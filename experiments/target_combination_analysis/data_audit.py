"""Read-only TRAIN/VALID audit. Never prepare data or open TEST/full_filtered."""
import argparse
import csv
from collections import deque
import json
from pathlib import Path
import yaml
from .common import HERE,ROOT,TARGETS,config,sha,write,frozen_checks,source_digest
from .safety import DataAccessGuard

def summary_checks(summary):
    cfg=config()
    assert summary['identity_hash']==cfg['fingerprint']
    assert summary['benchmark_filename']==['train','valid']
    assert summary['manifest_join_is_exact'] is True
    assert summary['forbidden_test_paths_loaded']==[]
    assert not summary['test_path_passed_to_search']
    for key in ['test_rows_in_inter_file','test_rows_in_benchmark_file']: assert summary[key]==0
    assert summary['rows']['test']==0 and summary['sequential_examples']['test']==0
    checks={}
    for path_key,hash_key in [('train_inter_path','train_inter_sha256'),('valid_inter_path','valid_inter_sha256'),('item_path','item_sha256'),('validation_source_row_ids_path','validation_source_row_ids_sha256')]:
        actual=sha(summary[path_key]);assert actual==summary['files'][hash_key],path_key
        checks[path_key]=actual
    return checks

def compare(row,current,history):
    # Compare raw IDs and labels, not floating-point RecBole remappings.
    assert [int(float(row[x])) for x in ['user_id','item_id','timestamp','source_row_id',*TARGETS]]==list(current)
    assert [int(x) for x in row['item_id_list'].split()]==[r[1] for r in history]
    assert [float(x) for x in row['timestamp_list'].split()]==[float(r[2]) for r in history]
    assert int(float(row['item_length']))==len(history)
    assert current[3] not in {r[3] for r in history}

def inter_rows(path):
    with Path(path).open() as f:
        header=f.readline().rstrip('\n').split('\t')
        names=[v.split(':')[0] for v in header]
        assert names==['user_id','item_id','timestamp','source_row_id',*TARGETS,'item_id_list','timestamp_list','item_length']
        for row in csv.reader(f,delimiter='\t'): yield dict(zip(names,row))

def run_audit():
    import polars as pl
    from experiments.stage3_auxiliary_analysis import target_audit as old
    frozen_checks();cfg=config()
    guard=DataAccessGuard(['/home/daryumin/iberdov/diplom/data',cfg['source']['validation_only_recbole_dir']]);guard.install()
    opt=yaml.safe_load((ROOT/cfg['source']['validation_only_config']).read_text())
    summary=json.loads(Path(opt['validation_only_data']['summary_json']).read_text())
    checks=summary_checks(summary)
    manifest=json.loads((ROOT/'outputs/data/protocol_b_multitask_manifest.json').read_text())
    assert manifest['join_diagnostics']['join_is_exact']
    assert manifest['dataset_fingerprint']['identity_hash_user_item_timestamp_split']==cfg['fingerprint']
    source=Path(cfg['source']['protocol_b_multitask_dir']);files={f['relative_path']:f['sha256'] for f in manifest['files']}
    for name in ['train.parquet','validation.parquet']:
        assert sha(source/name)==files[name]
    cols=['user_id','item_id','timestamp','source_row_id',*TARGETS]
    train=pl.read_parquet(source/'train.parquet',columns=cols).sort(['user_id','timestamp','source_row_id','item_id'])
    valid=pl.read_parquet(source/'validation.parquet',columns=cols).sort(['user_id','timestamp','source_row_id','item_id'])
    assert train.height==1086518 and valid.height==23951
    assert train['source_row_id'].n_unique()==train.height and valid['source_row_id'].n_unique()==valid.height
    assert not set(train['source_row_id'].to_list()) & set(valid['source_row_id'].to_list())
    for frame in [train,valid]:
        for t in TARGETS: assert frame[t].null_count()==0 and set(frame[t].unique().to_list())=={0,1}
    reader=iter(inter_rows(summary['train_inter_path']));tails={};history=deque(maxlen=50);user=None;count=0
    for row in train.iter_rows():
        row=tuple(int(v) for v in row)
        if user!=row[0]:
            if user is not None: tails[user]=list(history)
            history=deque(maxlen=50);user=row[0]
        if history:
            compare(next(reader),row,history);count+=1
        history.append(row)
    tails[user]=list(history)
    assert next(reader,None) is None and count==1062567
    reader=iter(inter_rows(summary['valid_inter_path']));ids=set()
    for row in valid.iter_rows():
        row=tuple(int(v) for v in row);history=tails[row[0]]
        assert (row[2],row[3],row[1])>(history[-1][2],history[-1][3],history[-1][1])
        compare(next(reader),row,history);ids.add(row[3])
    assert next(reader,None) is None and len(tails)==23951
    assert ids=={int(x) for x in Path(summary['validation_source_row_ids_path']).read_text().splitlines()}
    # Existing descriptive audit, redirected only to the sibling's artifacts.
    audit_cfg=yaml.safe_load((ROOT/'experiments/stage3_auxiliary_analysis/config.yaml').read_text())
    audit_cfg['outputs']={'runs_dir':str(HERE/'artifacts/audit'),'target_audit_run_id':'target_combo_target_audit_001'}
    audit_path=HERE/'artifacts/audit/target_combo_target_audit_001.json'
    if audit_path.exists(): raise FileExistsError(audit_path)
    descriptive=old.target_audit(audit_cfg)
    assert not descriptive['relationship_source'].startswith('missing:')
    for row in descriptive['target_audit']:
        if row['target'] in TARGETS:
            assert row['missing_rate_train']==0 and row['train_observations']==1086518
            assert row['train_positive_count']==int(train[row['target']].sum())
    payload={'status':'passed','source_digest':source_digest(),'fingerprint':cfg['fingerprint'],'file_checks':checks,'train_examples_verified':count,'validation_examples_verified':valid.height,'users_verified':len(tails),'label_join_verified_against_source_parquets':True,'history_exact_train_prefix':True,'current_source_row_excluded':True,'target_labels_absent_from_history_fields':True,'existing_target_audit_path':str(audit_path),'existing_target_audit_sha256':sha(audit_path),'test_evaluation_count':0,'data_access':guard.record()}
    write(HERE/'artifacts/data_audit.json',payload);print(json.dumps(payload,indent=2))

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();run_audit()
