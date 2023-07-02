import torch
from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
from torch.utils.data import DataLoader
from function import MySampler, empty_cache, make_training_set

# set environment variable, don't need to download the model every time
HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1

# get model and add pooling layer
ancient_chinese_model = models.Transformer("Jihuai/bert-ancient-chinese", max_seq_length=256)
pooling_layer = models.Pooling(ancient_chinese_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[ancient_chinese_model, pooling_layer])

# define loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# ------ Training Step 1: using parallels from ctext.org ------
# define training set
ctext_training_set = make_training_set('ctext_parallel_pair')
# define dataloader
ctext_dataloader = DataLoader(dataset=ctext_training_set, sampler=MySampler(ctext_training_set), batch_size=8)
# tune the model, output_path is the path to save the fine-tuned model
empty_cache()
model.fit(train_objectives=[(ctext_dataloader, train_loss)], epochs=3, use_amp=True,
          checkpoint_path='./checkpoint/ctext', output_path='./result/ctext', checkpoint_save_steps=10000)
empty_cache()

# ------ Training Step 2: using traditional-simplified Chinese character parallels ------
# define training set
train_ts_set = make_training_set('char_train')
# define dataloader
ts_dataloader = DataLoader(dataset=train_ts_set, batch_size=8)
# tune the model
empty_cache()
model.fit(train_objectives=[(ts_dataloader, train_loss)], epochs=3, use_amp=True,
          checkpoint_path='./checkpoint/trad_simple', output_path='./result/trad_simple', checkpoint_save_steps=10000)
empty_cache()

# ------ Training Step 3: using ancient-modern Chinese parallel corpus ------
# define training set
train_cam_set = make_training_set('cam_train')
# define dataloader
cam_dataloader = DataLoader(dataset=train_cam_set, batch_size=8)
# tune the model
empty_cache()
model.fit(train_objectives=[(cam_dataloader, train_loss)], epochs=2, use_amp=True,
          checkpoint_path='./checkpoint/cam_parallel', output_path='./result/cam_parallel', checkpoint_save_steps=10000)
empty_cache()
