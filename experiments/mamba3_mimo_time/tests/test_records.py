import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.mamba3_mimo_time import config as c, records as r
from experiments.mamba3_mimo_time.provenance import validate_ownership, require_gate
from experiments.mamba3_mimo_time.report import summarize, markdown


class RecordsTests(unittest.TestCase):
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('result.json','run.lock','reservation.json'):
                p=Path(d)/name
                r.create(p, {'owner':1})
                with self.assertRaises(FileExistsError):
                    r.create(p, {'owner':2})
                self.assertEqual(r.read(p), {'owner':1})
            p=Path(d)/'result.json'
            r.update(p, {'owner':1,'status':'PASS'})
            self.assertEqual(r.read(p)['status'],'PASS')

    def test_registry_all_required_leaves(self):
        specs=c.plan()['required_cases']
        rows=[dict(case_id=s['id'],status='PASS',required_keys=s['required_keys'],
                   checks={k:dict(passed=True) for k in s['required_keys']}) for s in specs]
        self.assertTrue(r.accepted_cases(rows,specs))
        for changed in (rows[:-1],rows+rows[-1:],list(reversed(rows))):
            self.assertFalse(r.accepted_cases(changed,specs))
        for mutation in ('missing','hidden','nan','unexpected','skip'):
            altered=copy.deepcopy(rows)
            k=next(iter(altered[0]['checks']))
            if mutation=='missing':
                altered[0]['checks'].pop(k);altered[0]['required_keys']=[]
            elif mutation=='hidden':
                altered[0]['checks'][k]['tensors']={'D':{'passed':False}}
            elif mutation=='nan':
                altered[0]['checks'][k]['error']=float('nan')
            elif mutation=='skip':
                altered[0]['checks'][k]['status']='SKIP'
            else:
                altered[0]['checks']['extra']={'passed':True}
            self.assertFalse(r.accepted_cases(altered,specs),mutation)

    def test_failed_case_persisted_and_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json';value={'status':'RUNNING'}
            r.create(p,value)
            spec={'id':'x','required_keys':['output','gradient']}
            registry=r.Registry(p,value,[spec])
            with self.assertRaises(RuntimeError):
                registry.run(spec, lambda save:r.case({'output':{'passed':True}}))
            self.assertEqual(r.read(p)['cases'][0]['missing_keys'],['gradient'])
            with patch.object(c,'GATE',p), self.assertRaises(ValueError):
                require_gate({})

    def test_failed_gate_stops_pipeline_before_smoke_or_fit(self):
        from experiments.mamba3_mimo_time import pipeline, report
        with tempfile.TemporaryDirectory() as d:
            log=Path(d);calls=[]
            def child(stage,*args):
                calls.append(stage)
                if stage=='admission':
                    raise RuntimeError('required nested FAIL')
            with patch.object(c,'LOGS',log), patch.object(c,'PIPELINE',log/'pipeline.json'), \
                 patch.object(pipeline,'identity',return_value={}), patch.object(pipeline,'child',side_effect=child), \
                 patch.object(pipeline.signal,'signal'), patch.dict(pipeline.os.environ,{},clear=False), \
                 patch.object(report,'write',return_value={'status':'INCOMPLETE','scientific_fits_started':0,'scientific_fits_completed':0}):
                self.assertEqual(pipeline.main(),1)
            self.assertEqual(calls,['preflight','admission'])
            self.assertEqual(r.read(log/'pipeline.json')['scientific_fits'],0)

    def test_serialized_ownership_and_submit_race(self):
        expected={'execution_commit':'a'*40,'TEST':'NOT_RUN','test_evaluation_count':0}
        login=dict(expected,status='PASS',tracked_clean=True,historical_sources_verified=True,published_commit='a'*40)
        reservation=dict(expected,status='RESERVED',login_sha256='abc',max_scientific_fits=3,token='b'*32)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            r.create(p/'login',login);r.create(p/'reservation',reservation)
            login,reservation=r.read(p/'login'),r.read(p/'reservation')
            validate_ownership(login,reservation,'abc',expected,'123')
            operational={'job_id':'123','token':'b'*32}
            r.create(p/'submitted',operational)
            validate_ownership(login,reservation,'abc',expected,'123',r.read(p/'submitted'))
            for field,value in [('job_id','124'),('token','c'*32)]:
                bad=dict(operational,**{field:value})
                with self.assertRaises(ValueError):
                    validate_ownership(login,reservation,'abc',expected,'123',bad)
            reservation['max_scientific_fits']=4
            with self.assertRaises(ValueError):
                validate_ownership(login,reservation,'abc',expected,'123')


def successful(mode, score):
    metrics={f'{m}@{k}':score for m in ('hit','recall','ndcg') for k in (5,10,20,50)}
    history=[dict(epoch=i,valid_ndcg10=score,valid_metrics=metrics,diagnostics={'epoch':i}) for i in range(2)]
    return dict(mode=mode,status='PASS',TEST='NOT_RUN',test_evaluation_count=0,scientific_fit_started=True,
                parameter_count=c.COUNTS[mode],actual_epochs=2,history=history,best_epoch=1,
                best_valid_metrics=metrics,best_valid_score=score,best_diagnostics=history[1]['diagnostics'],
                train_seconds=1,valid_seconds=2,peak_gpu_allocated_bytes=1024,peak_gpu_reserved_bytes=2048)


class ReportTests(unittest.TestCase):
    def test_negative_delta_partial_and_test_separation(self):
        records={'base':successful('base',.07),'dual':successful('dual',.06)}
        s=summarize(json.loads(json.dumps(records)))
        self.assertEqual(s['status'],'INCOMPLETE')
        self.assertAlmostEqual(s['comparisons'][0]['absolute'],-.01)
        self.assertEqual(s['rows'][2]['status'],'NOT_RUN')
        self.assertIsNone(s['rows'][2]['metrics'])
        self.assertFalse(s['rows'][0]['first27_complete'])
        self.assertIn('-0.0100',markdown(s))
        self.assertIn('TEST не использовался',markdown(s))
        records['base']['TEST']='PASS'
        with self.assertRaises(ValueError):
            summarize(records)

    def test_last_tie_and_missing_epoch(self):
        r=successful('base',.06);r['best_epoch']=0
        with self.assertRaises(ValueError):
            summarize({'base':r})
        r=successful('base',.06);r['history'][1]['epoch']=2
        with self.assertRaises(ValueError):
            summarize({'base':r})
