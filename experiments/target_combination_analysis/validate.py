"""Deterministic local/static checks; no data or GPU required."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from .common import HERE,ROOT,frozen_checks,source_digest,write

def main():
    checks=frozen_checks()
    for path in HERE.rglob('*.py'):
        if 'runtime' not in path.parts:ast.parse(path.read_text(),filename=str(path))
    for script in (ROOT/'slurm').glob('*target_combinations*.sh'):subprocess.run(['bash','-n',str(script)],check=True)
    with tempfile.TemporaryDirectory() as folder:
        xml=Path(folder)/'junit.xml'
        r=subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q','--junitxml='+str(xml)],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        cases=[{'name':x.attrib['name'],'passed':not any(y.tag in ['error','failure','skipped'] for y in x)} for x in ET.parse(xml).iter('testcase')]
    passed=r.returncode==0 and len(cases)>=23 and all(x['passed'] for x in cases)
    write(HERE/'verification.json',{'status':'passed' if passed else 'failed','source_digest':source_digest(),'cases':cases,'test_count':len(cases),'dataset_loaded':False,**checks})
    if not passed:raise SystemExit(1)
    print(f'Passed {len(cases)}/{len(cases)}; source-bound certificate written')

if __name__=='__main__':main()
