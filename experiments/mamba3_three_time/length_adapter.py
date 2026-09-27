"""MIMO kernel-boundary zero tail; never pads item histories or model states."""

import torch

AXES = dict(q=1, k=1, v=1, z=1, angles=1, adt=2, dw=2, dp=2, trap=2)


def prepare(values, *, chunk=8, extra_chunks=0):
    if chunk != 8 or extra_chunks not in (0, 1):
        raise ValueError("Frozen rank4/chunk8 diagnostic scope")
    length = values["q"].shape[1]
    if length < 1 or values["q"].shape[2] != 4:
        raise ValueError("Nonempty rank4 inputs required")
    target = ((length + chunk - 1) // chunk + extra_chunks) * chunk
    padded, aliases = {}, {}
    for key, value in values.items():
        if key not in AXES or value is None:
            padded[key] = value
            continue
        axis = AXES[key]
        if value.shape[axis] != length:
            raise ValueError("Sequence axis mismatch: " + key)
        identity = (id(value), axis)
        if identity not in aliases:
            shape = list(value.shape)
            shape[axis] = target - length
            aliases[identity] = (value if target == length else
                                torch.cat((value, value.new_zeros(shape)), dim=axis))
        padded[key] = aliases[identity]
    return padded, dict(input_length=length, kernel_length=target, chunk_size=chunk,
                        extra_chunks=extra_chunks, axes=AXES, tail_value=0)


def official(values, *, native=False, extra_chunks=0):
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
    if values["dw"] is not values["dp"]:
        raise ValueError("Official comparator requires tied write/phase")
    x, record = (values, {"input_length": values["q"].shape[1]}) if native else prepare(values, extra_chunks=extra_chunks)
    if native and x["q"].shape[1] % 8:
        raise ValueError("Native backward requires aligned length")
    y = mamba3_mimo(x["q"], x["k"], x["v"], x["adt"], x["dw"], x["trap"],
        x["qb"], x["kb"], x["mv"], x["mz"], x["mo"], x["angles"], x["d"], x["z"],
        8, 4, x["v"].dtype)
    return y[:, :record["input_length"]]


def official_mixer(mixer, inputs):
    """Unmodified pinned module forward, with only its kernel boundary wrapped."""
    import types
    function = type(mixer).forward
    def boundary(**kw):
        if (kw["chunk_size"] != 8 or kw["rotary_dim_divisor"] != 4 or kw["return_state"] or
                kw["cu_seqlens"] is not None or kw["fuse_pregate_headwise_rms_norm"] or
                kw["outproj_norm_weight"] is not None):
            raise ValueError("Unsupported official length-adapter scope")
        return official(dict(q=kw["Q"],k=kw["K"],v=kw["V"],adt=kw["ADT"],
            dw=kw["DT"],dp=kw["DT"],trap=kw["Trap"],qb=kw["Q_bias"],kb=kw["K_bias"],
            mv=kw["MIMO_V"],mz=kw["MIMO_Z"],mo=kw["MIMO_Out"],angles=kw["Angles"],d=kw["D"],z=kw["Z"]))
    namespace = dict(function.__globals__,mamba3_mimo_combined=boundary)
    local = types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__)
    local.__kwdefaults__ = function.__kwdefaults__
    return local(mixer,inputs)
