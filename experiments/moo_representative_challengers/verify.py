"""Execute required unit/parity gates and record a source-hash-bound certificate."""
import importlib.metadata
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from .common import HERE, ROOT, git, source_digest, verify_historical_inputs, write_json


def main():
    started=time.monotonic()
    historical=verify_historical_inputs()
    with tempfile.TemporaryDirectory(prefix='moo_verification_') as temporary:
        xml=temporary+'/junit.xml'
        result=subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q','--junitxml='+xml],cwd=ROOT,
                              env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        suite=ET.parse(xml).getroot()
        cases=[{'name':c.attrib['name'],'class':c.attrib.get('classname'),'seconds':float(c.attrib['time']),
                'passed':not any(e.tag in ['failure','error','skipped'] for e in c)} for c in suite.iter('testcase')]
    passed=result.returncode==0 and len(cases)>=24 and all(c['passed'] for c in cases)
    report={'status':'passed' if passed else 'failed','source_digest':source_digest(),
            'git_commit_at_verification':git('rev-parse','HEAD'),'git_branch':git('branch','--show-current'),
            'test_evaluation_count':0,'dataset_loaded':False,'historical_inputs':historical,
            'cases':cases,'test_count':len(cases),'runtime_seconds':time.monotonic()-started,
            'python':platform.python_version(),
            'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    write_json(HERE/'verification.json',report)
    if not passed: raise SystemExit(1)
    print('All mandatory unit/parity gates passed; certificate bound to source/config hashes')


if __name__=='__main__': main()
