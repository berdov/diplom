import copy
import json
import math
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch

from experiments.mamba3_head_timescales.confirmation import config as c,report,provenance,pipeline,submit,continuation
from experiments.mamba3_head_timescales.process_env import child_environment
from experiments.mamba3_head_timescales.progress import progress_callback
from experiments.mamba3_mimo_time.records import create,read,sha,Registry,accepted_cases,case

PILOTS={v:read(c.pilot_path(v)) for v in c.MODES}
OLD_PLAN=read(c.PILOT_ROOT/'study_plan.json')


@contextmanager
def scratch():
    with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
        root=Path(directory);pilot=root/'experiments/mamba3_head_timescales';here=pilot/'confirmation'
        for name,value in dict(ROOT=root,PILOT_ROOT=pilot,HERE=here,RUNTIME=here/'runtime',LAUNCHER=root/'slurm/confirmation.sh').items():
            stack.enter_context(patch.object(c,name,value))
        create(pilot/'study_plan.json',OLD_PLAN)
        for v,r in PILOTS.items():create(c.pilot_path(v),r)
        create(here/'study_plan.json',c.expected_plan())
        create(here/'source_index.json',dict(study_id=c.STUDY,allocation_attempt='001',entries=[dict(t,attempt='001',result=str(c.paths(t['variant'],t['seed'])['result'].relative_to(root))) for t in c.tasks()]))
        yield root


def successful(variant='fixed',seed=2027,score=.06,epochs=39,base=None):
    r=copy.deepcopy(PILOTS[variant]);best=epochs-12
    row=copy.deepcopy(r['history'][0]);history=[]
    for epoch in range(epochs):
        x=copy.deepcopy(row);x['epoch']=epoch
        x['valid_ndcg10']=score if epoch==best else score-.001
        x['valid_metrics']['ndcg@10']=x['valid_ndcg10'];history.append(x)
    r.update(run_id=c.paths(variant,seed)['run_id'],seed=seed,history=history,actual_epochs=epochs,best_epoch=best,
             best_valid_score=score,best_valid_metrics=history[best]['valid_metrics'],best_diagnostics=history[best]['diagnostics'],
             first27_complete=epochs>=27,first27_best_ndcg10=max(x['valid_ndcg10'] for x in history[:27]) if epochs>=27 else None,
             execution_attempt='001',job_id='123')
    r['config']['seed']=r['effective_config']['seed']=seed
    if base:r.update(base)
    return r


class RecordsTests(unittest.TestCase):
    def test_plan_order_and_seed2026_blocked(self):
        self.assertEqual([(t['seed'],t['variant']) for t in c.tasks()],[(s,v) for s in c.SEEDS for v in c.MODES])
        with self.assertRaises(ValueError):c.paths('fixed',2026)
        with self.assertRaises(ValueError):c.settings('fixed',2026,checkpoint_dir='/tmp/fixture')
        with self.assertRaises(ValueError):c.allocation('003')

    def test_settings_only_seed_and_output(self):
        for seed in c.SEEDS:
            for variant in c.MODES:
                values=c.settings(variant,seed);old=PILOTS[variant]['config']
                self.assertEqual({k for k in set(values)|set(old) if values.get(k)!=old.get(k)},{'seed','checkpoint_dir'})
                self.assertEqual(values['seed'],seed)

    def test_index_duplicate_and_current_preserved_rejected(self):
        with scratch():
            path=c.allocation()['source_index'];original=read(path)
            self.assertEqual(len(c.index()['entries']),12)
            for damage in ('duplicate','preserved'):
                broken=copy.deepcopy(original)
                if damage=='duplicate':broken['entries'][1]=copy.deepcopy(broken['entries'][0])
                else:broken['entries'][0]['preserved']={'fake':True}
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):c.index()

    def test_sample_std_empty_single_and_negative_pairs(self):
        self.assertEqual(report.stats([])['mean'],None)
        self.assertEqual(report.stats([1])['sample_std'],None)
        records={}
        for seed,delta in zip(c.SEEDS,[-.002,-.001,0,.003]):
            for v,score in [('fixed',.05),('shared_tau',.06),('head_tau',.06+delta)]:records[seed,v]=successful(v,seed,score)
        s=report.summarize(records)['cohorts']['new4_full']['contrasts']['head_tau-shared_tau']
        self.assertAlmostEqual(s['mean'],0)
        self.assertAlmostEqual(s['sample_std'],math.sqrt(14e-6/3))
        self.assertEqual((s['positive'],s['zero'],s['negative']),(1,1,2))

    def test_first27_pair_does_not_require_third(self):
        records={(2027,v):successful(v,2027,epochs=26 if v=='fixed' else 39) for v in c.MODES}
        s=report.summarize(records)
        first=s['cohorts']['new4_first27']
        self.assertEqual(first['complete_triples'],[])
        self.assertEqual(first['contrasts']['head_tau-shared_tau']['seeds'],[2027])
        self.assertEqual(first['contrasts']['head_tau-fixed']['n_available'],0)
        self.assertEqual(s['rows'][3]['status'],'PASS')
        self.assertIsNone(s['rows'][3]['first27_ndcg10'])

    def test_window26_vs27(self):
        for count,expected in [(26,False),(27,True)]:
            r=successful(epochs=count)
            s=report.summarize({(2027,'fixed'):r})['rows'][3]
            self.assertEqual(s['first27_complete'],expected)
            self.assertEqual(s['first27_ndcg10'] is not None,expected)

    def test_missing_first_and_last_result(self):
        records={(t['seed'],t['variant']):successful(t['variant'],t['seed']) for t in c.tasks()}
        del records[2027,'fixed'];del records[2030,'head_tau']
        s=report.summarize(records,PILOTS)
        self.assertEqual(s['status'],'INCOMPLETE')
        self.assertEqual(len(s['rows']),15)
        self.assertEqual(s['scientific_fits_completed'],10)
        self.assertEqual(s['rows'][3]['status'],'NOT_RUN');self.assertEqual(s['rows'][-1]['status'],'NOT_RUN')

    def test_pair_mean_uses_same_seed_intersection(self):
        records={(2027,'head_tau'):successful('head_tau',2027,.06),(2027,'shared_tau'):successful('shared_tau',2027,.05),
                 (2028,'head_tau'):successful('head_tau',2028,.1)}
        s=report.summarize(records)['cohorts']['new4_full']['contrasts']['head_tau-shared_tau']
        self.assertEqual(s['seeds'],[2027]);self.assertAlmostEqual(s['relative_percent'],20)

    def test_failed_metrics_not_aggregated(self):
        r=successful();r['status']='FAIL'
        s=report.summarize({(2027,'fixed'):r})
        self.assertIsNone(s['rows'][3]['ndcg10']);self.assertEqual(s['scientific_fits_completed'],0)

    def test_selection_and_metric_schema_fail_closed(self):
        for kind in ('missing','different'):
            r=successful()
            if kind=='missing':del r['history'][0]['valid_metrics']['hit@5']
            else:r['history'][0]['valid_metrics']['ndcg@10']+=.1
            self.assertEqual(report.summarize({(2027,'fixed'):r})['rows'][3]['status'],'FAIL')

    def test_history_gap_duplicate_and_malformed_partial(self):
        for damage in ('gap','duplicate','missing','none'):
            r=successful()
            if damage=='gap':r['history'][2]['epoch']=4
            elif damage=='duplicate':r['history'][2]['epoch']=1
            elif damage=='missing':del r['history'][2]['epoch']
            else:r['history']=None
            s=report.summarize({(2027,'fixed'):r})
            self.assertEqual(s['rows'][3]['status'],'FAIL');self.assertIsNone(s['rows'][3]['ndcg10'])

    def test_last_tie_and_earlystop(self):
        r=successful(epochs=39);r['history'][26]['valid_ndcg10']=r['best_valid_score']
        r['history'][26]['valid_metrics']['ndcg@10']=r['best_valid_score'];r['first27_best_ndcg10']=r['best_valid_score']
        report.validate_record(r,c.tasks()[0])
        r['best_epoch']=26
        with self.assertRaises(ValueError):report.validate_record(r,c.tasks()[0])
        r=successful();r['history'].pop();r['actual_epochs']-=1
        with self.assertRaises(ValueError):report.validate_record(r,c.tasks()[0])

    def test_pairing_rejects_serialized_batch_or_common_drift(self):
        records={(2027,v):successful(v) for v in c.MODES}
        self.assertEqual(report.summarize(records)['cohorts']['new4_full']['complete_triples'],[2027])
        for key in ('first_train_batch_sha256','initial_backbone_sha256','initial_common_calibrator_hashes','rng_components'):
            damaged=copy.deepcopy(records);damaged[2027,'head_tau'][key]='bad'
            s=report.summarize(damaged)
            self.assertNotEqual(s['rows'][5]['status'],'PASS')

    def test_partial_write_bad_json_keeps_numeric_markdown(self):
        with scratch():
            a=c.allocation();p=c.paths('fixed',2027)['result'];p.parent.mkdir(parents=True);p.write_text('{broken')
            s=report.write(dict(execution_attempt='001'),'001','serialization fixture')
            self.assertEqual(s['status'],'INCOMPLETE');self.assertEqual(s['scientific_fits_start_unknown'],1)
            self.assertTrue(a['summary'].exists());self.assertTrue(a['summary'].with_suffix('.md').exists())
            self.assertEqual(p.read_text(),'{broken')
            self.assertEqual(read(c.paths('head_tau',2030)['result'])['status'],'NOT_RUN')

    def test_retry_reason_not_blocking_reason_and_no_overwrite(self):
        with scratch():
            base=dict(execution_attempt='001',retry_reason='old serialization')
            s=report.write(base,'001')
            self.assertEqual(s['retry_reason'],'old serialization');self.assertIsNone(s['blocking_reason'])
            with self.assertRaises(FileExistsError):report.write(base,'001')

    def test_partial_json_scalar_keeps_summary(self):
        for encoded in ('null','123','"fixture"'):
            with scratch():
                p=c.paths('fixed',2027)['result'];p.parent.mkdir(parents=True);p.write_text(encoded)
                s=report.write(dict(execution_attempt='001'),'001')
                self.assertEqual(s['scientific_fits_start_unknown'],1)
                self.assertEqual(s['rows'][3]['status'],'FAIL')
                self.assertTrue(c.allocation()['summary'].with_suffix('.md').exists())

    def test_already_failed_malformed_history_keeps_summary(self):
        with scratch():
            p=c.paths('fixed',2027)['result']
            r=successful();r.update(status='FAIL',history=[{}]*30)
            create(p,r)
            s=report.write(dict(execution_attempt='001'),'001','original failure')
            self.assertEqual(s['rows'][3]['status'],'FAIL');self.assertIsNone(s['rows'][3]['ndcg10'])
            self.assertTrue(c.allocation()['summary'].with_suffix('.md').exists())

    def test_source_previous_variant_same_seed_and_order(self):
        with scratch():
            base=dict(execution_attempt='001',job_id='123')
            for v in c.MODES:create(c.paths(v,2027)['result'],successful(v,base=base))
            previous=report.previous_records('head_tau',2027,'001',base)
            self.assertEqual([r['time_scale_mode'] for r in previous],['fixed','shared_tau'])
            self.assertEqual(report.previous_records('fixed',2028,'001',base),[])
            with self.assertRaises(FileNotFoundError):report.previous_records('head_tau',2028,'001',base)

    def test_serialized_ownership_limits(self):
        expected=dict(execution_attempt='001',execution_commit='a'*40,source_hash='b'*64)
        login=dict(expected,status='PASS',tracked_clean=True,source_blobs_verified=True,published_commit='a'*40)
        reservation=dict(expected,status='RESERVED',login_sha256='c'*64,token='d'*32,max_scientific_fits=12,
                         tasks=c.tasks(),planned_tasks=c.tasks(),jobs_requested=1,allocation_seconds=28800)
        login,reservation=json.loads(json.dumps([login,reservation]))
        provenance.validate_ownership(login,reservation,'c'*64,expected,'123',dict(job_id=None,token='d'*32,status='SUBMISSION_UNKNOWN_NO_RETRY'))
        for key,value in [('max_scientific_fits',13),('tasks',[]),('allocation_seconds',99999),('source_hash','foreign')]:
            with self.assertRaises(ValueError):provenance.validate_ownership(login,dict(reservation,**{key:value}),'c'*64,expected,'123')

    def test_submit_intent_precedes_sbatch_and_duplicate_blocked(self):
        with scratch(),patch.object(submit,'bindings',return_value=dict(execution_commit='a'*40)):
            a=c.allocation()
            def fake(command,**kwargs):
                self.assertTrue(a['reservation'].exists());self.assertTrue(a['submission'].exists())
                self.assertEqual(read(a['submission'])['status'],'SUBMISSION_UNKNOWN_NO_RETRY')
                self.assertIn('--time=08:00:00',command)
                return subprocess.CompletedProcess(command,0,'123;fixture\n','')
            with patch.object(submit.subprocess,'run',side_effect=fake) as call:
                r=submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(r['job_id'],'123')
                with self.assertRaises(FileExistsError):submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(call.call_count,1)

    def test_ambiguous_submit_durable_no_retry(self):
        with scratch(),patch.object(submit,'bindings',return_value=dict(execution_commit='a'*40)):
            with patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'ambiguous','')) as call:
                with self.assertRaises(RuntimeError):submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(read(c.allocation()['submission'])['status'],'SUBMISSION_UNKNOWN_NO_RETRY')
                with self.assertRaises(FileExistsError):submit.reserve_and_submit(dict(execution_commit='a'*40),dict(source_hash='b'*64))
                self.assertEqual(call.call_count,1)

    def test_real_progress_registry_empty_not_pass_negative_preserved(self):
        spec=dict(id='causal',required_keys=['a','b'])
        for negative in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'gate.json';value={};create(p,value);registry=Registry(p,value,[spec])
                def execute(save):
                    callback=progress_callback(save,{})
                    callback({},dict(event='gradient_forward_started'))
                    observed=read(p)['cases'][0]
                    self.assertEqual(observed['status'],'RUNNING');self.assertEqual(observed['checks'],{})
                    leaves={'a':{'passed':not negative},'b':{'passed':True}}
                    callback(leaves,dict(event='complete'))
                    return case(leaves)
                if negative:
                    with self.assertRaises(RuntimeError):registry.run(spec,execute)
                    self.assertFalse(read(p)['cases'][0]['checks']['a']['passed'])
                else:
                    registry.run(spec,execute);self.assertTrue(accepted_cases(read(p)['cases'],[spec]))

    def test_empty_or_missing_final_checks_cannot_pass(self):
        for checks in ({},{'a':{'passed':True}}):
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'gate.json';value={};create(p,value);spec=dict(id='x',required_keys=['a','b']);reg=Registry(p,value,[spec])
                with self.assertRaises(RuntimeError):reg.run(spec,lambda save:dict(checks=checks,required_keys=sorted(checks)))
                self.assertEqual(read(p)['cases'][0]['status'],'FAIL')

    def test_pipeline_failure_keeps_real_partial_summary(self):
        with scratch(),patch.object(pipeline,'identity',return_value=dict(execution_attempt='001',execution_commit='a'*40,source_hash='b'*64,job_id='123')), \
             patch.object(pipeline,'inherited',side_effect=RuntimeError('infrastructure fixture')),patch.object(pipeline.signal,'signal'):
            self.assertEqual(pipeline.main(),1)
            a=c.allocation();self.assertEqual(read(a['summary'])['scientific_fits_completed'],0)
            self.assertTrue(a['summary'].with_suffix('.md').exists())
            self.assertIn('infrastructure fixture',read(a['pipeline'])['traceback'])
            self.assertEqual(read(c.paths('head_tau',2030)['result'])['status'],'NOT_RUN')

    def test_continuation_never_restarts_started_or_unknown(self):
        records=[dict(status='NOT_RUN',scientific_fit_started=False,history=[]) for _ in c.tasks()]
        self.assertEqual(len(continuation.eligible_records(records)),12)
        for damage in (dict(scientific_fit_started=True),dict(scientific_fit_started=None),dict(status='FAIL',error='Nonfinite gradient')):
            bad=copy.deepcopy(records);bad[0].update(damage)
            with self.assertRaises(ValueError):continuation.eligible_records(bad)

    def test_cpu_mask_and_gpu_mask_preserved(self):
        parent=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-fixture')
        self.assertEqual(child_environment('preflight',parent)['CUDA_VISIBLE_DEVICES'],'')
        for variant in c.MODES:self.assertEqual(child_environment(variant,parent)['CUDA_VISIBLE_DEVICES'],'GPU-fixture')
        self.assertEqual(parent['CUDA_VISIBLE_DEVICES'],'GPU-fixture')

    def test_handoff_phase_does_not_regress_or_lose_job(self):
        from experiments.mamba3_head_timescales.confirmation.handoff import save
        with scratch():
            save('pipeline_finished',job_ids=['123'],next_safe_step='audit')
            s=save('submitted',job_ids=['124'],next_safe_step='monitor')
            self.assertEqual(s['phase'],'pipeline_finished');self.assertEqual(s['job_ids'],['123','124'])
            self.assertEqual(s['next_safe_step'],'audit')
            save('audited',execution_attempt='001')
            s=save('submitted',execution_attempt='002',job_ids=['125'],next_safe_step='monitor002')
            self.assertEqual(s['phase'],'submitted');self.assertEqual(s['next_safe_step'],'monitor002')


if __name__=='__main__':unittest.main()
