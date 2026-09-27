# Аудит pinned Mamba3: forward и backward

Источник: [state-spaces/mamba, e9594ce1c732d97440f0332fdc43170a2294dbfa](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa).
SHA256 каждого использованного файла: [manifest](../upstream_manifest.json).
Это аудит точного pin, не текущего upstream main.

```text
u -> in_proj -> x/z/B/C/dd_dt/dd_A/Trap/Angles
B,C -> RMSNorm -> kernel: bias -> rotary -> recurrence/readout
d=softplus(dd_dt+dt_bias); A=-heavy_tail(dd_A), clamp(max=-A_floor)
                   +-> ADT=A*(d*s_decay) -> exp/log-prefix decay -> dADT
gap -> calibrators +-> DT_write=d*s_write -> gamma/beta weights -> dDT_write
                   +-> DT_phase=d*s_phase -> angle_dt_fwd -> dTheta
                                                   angle_dt_bwd -> dDT_phase,dAngles
Trap -> sigmoid -> gamma/beta -> dTrap (including sigmoid derivative)
Angles -> tanh_approx*PI -> multiply DT_phase -> cumsum, mod 2PI -> rotary
state/readout -> D skip -> SiLU(Z) -> output; MIMO includes rank projections
```

| Вход | Форма / dtype в frozen модели | Forward consumer | Backward source |
|---|---|---|---|
| ADT | B,H,L / fp32 | prefix decay и recurrent state | SISO compute_dqkv; MIMO bwd_dadt_fused_triton |
| DT_write | B,H,L / fp32 | gamma[t]=DT[t]*sigmoid(Trap[t]); beta[t]=DT[t]*(1-sigmoid(Trap[t])) | SISO compute_ddt_dtrap_dinput_states; MIMO bwd_dtrap_ddt_triton |
| DT_phase | B,H,L / fp32 | angle_dt_fwd | angle_dt_bwd: сумма reverse-cumsum(dTheta)*tanh(raw)*PI |
| Trap | B,H,L / bf16 | sigmoid и write weights | write backward, затем sigmoid derivative |
| Angles | B,L,H,N/4 / SISO bf16, MIMO fp32 | tanh_approx, PI, накопление и mod | reverse-cumsum(dTheta)*DT_phase*PI*sech2_approx(raw) |

`d` и scales до permute имеют B,L,H. Все коэффициенты dimensionless в данной
модели; gap в ms используется только для calibrators с frozen TRAIN reference.
Это дополнительные управляющие функции одного gap, а не независимые процессы.

## Точные индексы и операции

- [modules/mamba3.py:150](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py#L150): split projection, negative A, softplus DT, B/C RMSNorm до kernel biases; SISO и MIMO ветки сохраняют layout и dtype.
- [angle_dt.py:94](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/angle_dt.py#L94): `tanh_approx(raw)*PI`, затем DT и cumsum; строки107-117: mod2PI текущего блока и carry. [Строка298](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/angle_dt.py#L298): reverse-cumsum в backward, DT/Angles grads раздельны.
- [mamba3_siso_fwd.py:322](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py#L322): kernel хранит source weight `gamma[t]+beta[t+1]`, где последний beta за концом равен0. Строго нижнетреугольный readout не использует будущие события; diagonal вычисляется отдельно через unrotated QK*gamma. Bias до rotary; SISO пары соседних координат, вне rotary angle0.
- Эквивалентная последовательная формула: `S[t]=exp(ADT[t])*(S[t-1]+beta[t]*KV[t-1])+gamma[t]*KV[t]`. Начальные S[-1],KV[-1]=0. ADT влияет также на вклад предыдущего входа. Chunk carry передаёт состояние; shifted weight на границе читает следующий DT, но не создаёт noncausal output. В reference используется отдельный unrotated diagonal, как в kernel.
- [mamba3_siso_bwd.py:1537](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py#L1537): dDT=(dGamma+dScale)*sigmoid(Trap)+dScale[t-1]*(1-sigmoid(Trap)); t0 не имеет предыдущего вклада. dTrap учитывает sigmoid derivative. [Combined wrapper](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py) прежде суммировал write/phase dDT; локальная версия возвращает их разным slots.
- [mamba3_mimo_fwd.py:196](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py#L196): те же shifted weights. Строки223-240: V*Psi и bias; строки269-279: MIMO rotary пары `(n,N/2+n)` для n<N/4, не SISO interleaved layout. Строки340-355: same-token R×R QK readout, D*projected V; строки382-412: SiLU(Z*Zeta), Phi и суммирование рангов.
- [mamba3_mimo_bwd.py:1343](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py#L1343): native backward возвращает Q/K/V, ADT, write-DT, Trap, biases, Psi/Zeta/Phi, dTheta, D/Z. [mamba3_mimo.py:230](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py#L230): angle_dt_bwd добавлял phase-DT в тот же DT; локальный wrapper возвращает отдельный input gradient.

SISO public wrapper принудительно приводит Q/K/V/Trap/Angles/Z к bf16.
SISO cumulative angles и saved states тоже bf16. MIMO имеет bf16 rank projections
внутри kernel, bf16 ряд промежуточных произведений и saved states, хотя parameters
и angle accumulation fp32. Поэтому полного строгого fp32 пути здесь нет.
Structural parity и независимый reference проверяются разными допусками.

Изменения локальной адаптации: только два DT slots и отдельные производные,
ограничение dense/no-cache/no-outnorm; kernels и установленный upstream не меняются.
Apache-2.0/copyright сохранены; reference написан отдельно через обычный PyTorch.
