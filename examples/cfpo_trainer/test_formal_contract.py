import copy
import unittest
from formal_contract import EXPECTED,validate


def valid_config():
    config={}
    for path,value in EXPECTED.items():
        parent=config; keys=path.split('.')
        for key in keys[:-1]: parent=parent.setdefault(key,{})
        parent[keys[-1]]=value
    config['trainer'].update(logger=['console','swanlab'],rollout_data_dir='/run/train',validation_data_dir='/run/val')
    config['actor_rollout_ref']['actor']['optim']['betas']=[.9,.999]
    config['actor_rollout_ref']['model']={'path':'/models/Qwen3-VL-2B-Instruct'}
    return config


class ContractTests(unittest.TestCase):
    def test_aligned_configuration(self):
        report=validate(valid_config())
        self.assertEqual(report['responses_per_step'],1920)
        self.assertEqual(report['responses_per_actor_minibatch'],640)

    def test_material_migration_errors_rejected(self):
        cases=[('trainer.total_training_steps',20),('trainer.n_gpus_per_node',1),
               ('data.train_batch_size',4),('trainer.test_freq',-1),
               ('actor_rollout_ref.actor.optim.optimizer','AdamW'),
               ('actor_rollout_ref.actor.clip_ratio_high',.2),
               ('actor_rollout_ref.rollout.response_length',1024),
               ('actor_rollout_ref.rollout.max_model_len',4096),
               ('actor_rollout_ref.rollout.max_num_batched_tokens',4096),
               ('actor_rollout_ref.rollout.ignore_eos',True),
               ('actor_rollout_ref.rollout.top_k',20),
               ('actor_rollout_ref.actor.optim.lr_warmup_steps',1),
               ('actor_rollout_ref.actor.policy_loss.loss_mode','gspo'),
               ('actor_rollout_ref.actor.fsdp_config.model_dtype','fp32')]
        for path,value in cases:
            with self.subTest(path=path):
                config=copy.deepcopy(valid_config()); parent=config; keys=path.split('.')
                for key in keys[:-1]: parent=parent[key]
                parent[keys[-1]]=value
                with self.assertRaises(ValueError): validate(config)

    def test_missing_generation_logging_rejected(self):
        config=valid_config(); config['trainer']['validation_data_dir']=None
        with self.assertRaises(ValueError): validate(config)


if __name__=='__main__': unittest.main()
