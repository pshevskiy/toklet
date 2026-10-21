"""Exact test/model evidence parser mutations; never launches native work."""
from pathlib import Path
import ast,json,re,unittest
source=Path(__file__).with_name('driver.py').read_text();tree=ast.parse(source)
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='validate_gpu_output')
namespace=dict(re=re,json=json);exec(compile(ast.Module(body=[function],type_ignores=[]),'checked-parser','exec'),namespace)
validate=namespace['validate_gpu_output']
GOOD='''running 1 test
native_metal_model=Qwen2.5-0.5B-Instruct ids=[1410, 1492, 752] state=7/6/1 combined_admission_bytes=2000000000 passed
native_metal_model=Qwen2.5-1.5B-Instruct ids=[614, 264, 3405] state=7/6/1 combined_admission_bytes=3000000000 passed
test native::tests::metal_native_selected_models_match_direct_sessions ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 10 filtered out; finished in 0.1s
'''
class ParserChecks(unittest.TestCase):
 def test_requires_one_actual_named_test_and_both_exact_models(self):
  self.assertEqual(len(validate(GOOD)),2)
  invalid=[GOOD.replace('1 passed','0 passed'),GOOD.replace('1 passed','2 passed'),GOOD.replace('0 ignored','1 ignored'),
           GOOD.replace('metal_native_selected_models_match_direct_sessions','other_test'),
           GOOD.replace('Qwen2.5-1.5B-Instruct','Qwen2.5-0.5B-Instruct'),GOOD.replace('[614, 264, 3405]','[614, 264, 0]'),
           GOOD.replace('state=7/6/1','state=7/7/0'),GOOD.replace('3000000000','0'),GOOD.replace('3000000000','99999999999'),
           '\n'.join(line for line in GOOD.splitlines() if not line.startswith('native_metal_model=Qwen2.5-1.5B'))]
  for index,text in enumerate(invalid):
   with self.subTest(index=index),self.assertRaises(ValueError):validate(text)
if __name__=='__main__':unittest.main()
