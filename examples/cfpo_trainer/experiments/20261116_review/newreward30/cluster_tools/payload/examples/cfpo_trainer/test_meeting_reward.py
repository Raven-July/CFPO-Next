"""Test the meeting reward against the actual mathruler grader and source snippet."""
import importlib.util
import os
from pathlib import Path
import re
import unittest
from mathruler.grader import extract_boxed_content, grade_answer

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('meeting_reward',BASE/'reward_function/base.py')
reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)


class MeetingRewardTests(unittest.TestCase):
    def test_four_combinations_and_numeric_truth(self):
        cases=[(r'reasoning</think>\boxed{42}',42,1.,1.,1.),
               (r'reasoning</think>\boxed{41}',42,.1,0.,1.),
               (r'\boxed{42}',42,.9,1.,0.),
               (r'\boxed{41}',42,0.,0.,0.)]
        for text,truth,score,accuracy,format_score in cases:
            with self.subTest(text=text):
                actual=reward.compute_score('test',text,truth)
                self.assertEqual(actual,dict(score=score,overall=score,accuracy=accuracy,format=format_score))

    def test_empty_answer_never_calls_grader(self):
        from unittest.mock import patch
        for text in [r'</think>\boxed{}',r'\boxed{}']:
            with patch.object(reward,'grade_answer',side_effect=AssertionError('empty answer graded')):
                self.assertEqual(reward.compute_score('test',text,'')['accuracy'],0.)
        # mathruler0.1.0 returns the nonempty string "None" when no box exists.
        # Match the meeting snippet exactly; do not claim bool(answer) catches that sentinel.
        for text in ['',r'reasoning</think>no final answer']:
            self.assertEqual(extract_boxed_content(text),'None')
            self.assertEqual(reward.compute_score('test',text,'42')['accuracy'],0.)

    def test_actual_grader_fraction_zero_and_nested_box(self):
        for text,truth in [(r'</think>\boxed{\frac{1}{2}}','0.5'),(r'</think>\boxed{0}',0)]:
            with self.subTest(text=text):
                self.assertEqual(reward.compute_score('test',text,truth)['accuracy'],1.)

    def test_matches_exact_meeting_snippet(self):
        note=Path(os.environ['MEETING_REWARD_REFERENCE_FILE']) if os.environ.get('MEETING_REWARD_REFERENCE_FILE') else BASE.parents[2]/'审查/会议记录/2026-11-11-GRPO训练复盘.md'
        text=note.read_text(encoding='utf-8')
        snippet=text.split('```python',1)[1].split('```',1)[0]
        ns=dict(re=re,extract_boxed_content=extract_boxed_content,grade_answer=grade_answer)
        exec(compile(snippet,str(note),'exec'),ns)
        for text in [r'<think>x</think>\boxed{42}',r'x</think>\boxed{42}',r'\boxed{42}',
                     r'x< / think >\boxed{42}',r'\boxed{42}</think>',r'</think>\boxed{}',
                     r'x</think>y</think>\boxed{42}']:
            for truth in ['42',0]:
                with self.subTest(text=text,truth=truth):
                    self.assertEqual(reward.compute_score('test',text,truth),ns['compute_score']('test',text,truth))

    def test_invalid_weight_and_custom_weight(self):
        for weight in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):reward.compute_score('test','',42,format_weight=weight)
        self.assertAlmostEqual(reward.compute_score('test',r'</think>\boxed{41}',42,format_weight=.2)['score'],.2)


if __name__=='__main__':unittest.main()
