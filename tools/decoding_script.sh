# Baseline NMT model

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/baseline/baseline_1/model.ep35.iter13032.bleu34.94.npz nmt/models/baseline/baseline_1/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/baseline/test.txt.1 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/baseline/baseline_2/model.ep43.iter15928.bleu37.83.npz nmt/models/baseline/baseline_2/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/baseline/test.txt.2 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/baseline/baseline_3/model.ep45.iter16652.bleu35.59.npz nmt/models/baseline/baseline_3/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/baseline/test.txt.3 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/baseline/baseline_1/model.ep35.iter13032.bleu34.94.npz,nmt/models/baseline/baseline_2/model.ep43.iter15928.bleu37.83.npz,nmt/models/baseline/baseline_3/model.ep45.iter16652.bleu35.59.npz nmt/models/baseline/baseline_1/model.json,nmt/models/baseline/baseline_2/model.json,nmt/models/baseline/baseline_3/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/baseline/ensemble_test.txt 12

for x in nmt/models/baseline/*.txt*; do sed -i -r 's/ \@(.*?)\@ /\1/g' $x; sed -i -r 's/@ //g' $x; sed -i -r 's/ @//g'

# Multitask (Multi30K) Inception V3 

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_inceptionv3/InceptionV3_1/model.ep31.iter11584.bleu38.15.npz nmt/models/multitask_inceptionv3/InceptionV3_1/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_inceptionv3/test.txt.1 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_inceptionv3/InceptionV3_2/model.ep33.iter12308.bleu38.59.npz nmt/models/multitask_inceptionv3/InceptionV3_2/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_inceptionv3/test.txt.2 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_inceptionv3/InceptionV3_3/model.ep29.iter10860.bleu38.45.npz nmt/models/multitask_inceptionv3/InceptionV3_3/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_inceptionv3/test.txt.3 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_inceptionv3/InceptionV3_1/model.ep31.iter11584.bleu38.15.npz,nmt/models/multitask_inceptionv3/InceptionV3_2/model.ep33.iter12308.bleu38.59.npz,nmt/models/multitask_inceptionv3/InceptionV3_3/model.ep29.iter10860.bleu38.45.npz nmt/models/multitask_inceptionv3/InceptionV3_3/model.json,nmt/models/multitask_inceptionv3/InceptionV3_3/model.json,nmt/models/multitask_inceptionv3/InceptionV3_3/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_inceptionv3/ensemble_test.txt 12

for x in nmt/models/multitask_inceptionv3/*.txt*; do sed -i -r 's/ \@(.*?)\@ /\1/g' $x; sed -i -r 's/@ //g' $x; sed -i -r 's/ @//g'

# Multitask (COCO) Inception V3

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_with_COCO/multitask_COCO_1/model.ep36.iter13394.bleu37.51.npz nmt/models/multitask_with_COCO/multitask_COCO_1/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_with_COCO/test.txt.1 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_with_COCO/multitask_COCO_2/model.ep36.iter13394.bleu38.66.npz nmt/models/multitask_with_COCO/multitask_COCO_2/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_with_COCO/test.txt.2 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_with_COCO/multitask_COCO_3/model.ep34.iter12670.bleu38.14.npz nmt/models/multitask_with_COCO/multitask_COCO_3/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_with_COCO/test.txt.3 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_with_COCO/multitask_COCO_1/model.ep36.iter13394.bleu37.51.npz,nmt/models/multitask_with_COCO/multitask_COCO_2/model.ep36.iter13394.bleu38.66.npz,nmt/models/multitask_with_COCO/multitask_COCO_3/model.ep34.iter12670.bleu38.14.npz nmt/models/multitask_with_COCO/multitask_COCO_1/model.json,nmt/models/multitask_with_COCO/multitask_COCO_1/model.json,nmt/models/multitask_with_COCO/multitask_COCO_1/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_with_COCO/ensemble_test.txt 12

./test_decoding.sh nmt/data/wmt_task1/test.en nmt/models/multitask_inceptionv3/InceptionV3_1/model.ep31.iter11584.bleu38.15.npz,nmt/models/multitask_inceptionv3/InceptionV3_2/model.ep33.iter12308.bleu38.59.npz,nmt/models/multitask_inceptionv3/InceptionV3_3/model.ep29.iter10860.bleu38.45.npz,nmt/models/multitask_with_COCO/multitask_COCO_1/model.ep36.iter13394.bleu37.51.npz,nmt/models/multitask_with_COCO/multitask_COCO_2/model.ep36.iter13394.bleu38.66.npz,nmt/models/multitask_with_COCO/multitask_COCO_3/model.ep34.iter12670.bleu38.14.npz nmt/models/multitask_inceptionv3/InceptionV3_3/model.json,nmt/models/multitask_inceptionv3/InceptionV3_3/model.json,nmt/models/multitask_inceptionv3/InceptionV3_3/model.json,nmt/models/multitask_with_COCO/multitask_COCO_1/model.json,nmt/models/multitask_with_COCO/multitask_COCO_1/model.json,nmt/models/multitask_with_COCO/multitask_COCO_1/model.json nmt/data/wmt_task1/en_dict.json nmt/data/wmt_task1/de_dict.json nmt/models/multitask_m30K_coco_ensemble_test.txt 12

for x in nmt/models/multitask_with_COCO/*.txt*; do sed -i -r 's/ \@(.*?)\@ /\1/g' $x; sed -i -r 's/@ //g' $x; sed -i -r 's/ @//g'

# Sub-word NMT model
# Decode the sentences, followed by spm_decoding back to raw text

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_1/model.ep33.iter12308.bleu34.42.npz nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_1/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_1/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_1/test.txt.encoded > nmt/models/sentencepiece16_jointvocab/test.txt.1

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_2/model.ep31.iter11584.bleu35.22.npz nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_2/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_2/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_2/test.txt.encoded > nmt/models/sentencepiece16_jointvocab/test.txt.2

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_3/model.ep32.iter11946.bleu34.31.npz nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_3/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_3/test.txt.encoded > nmt/models/sentencepiece16_jointvocab/test.txt.3

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_3/model.ep32.iter11946.bleu34.31.npz,nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_2/model.ep31.iter11584.bleu35.22.npz,nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_3/model.ep32.iter11946.bleu34.31.npz nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_1/model.json,nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_2/model.json,nmt/models/sentencepiece16_jointvocab/multi30k_sentencepiece16k_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/sentencepiece16_jointvocab/ensemble_test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/sentencepiece16_jointvocab/ensemble_test.txt.encoded > nmt/models/sentencepiece16_jointvocab/ensemble_test.txt

# Sub-word CONCAT

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_ncT_CONCAT_sp16k_1/model.ep16.iter53776.bleu38.58.npz nmt/models/multi30kT_ncT_CONCAT_sp16k_1/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_ncT_CONCAT_sp16k_1/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_ncT_CONCAT_sp16k_1/test.txt.encoded > nmt/models/multi30kT_ncT_CONCAT_sp16k_1/test.txt.1

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_ncT_CONCAT_sp16k_2/model.ep15.iter50415.bleu39.23.npz nmt/models/multi30kT_ncT_CONCAT_sp16k_2/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_ncT_CONCAT_sp16k_2/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_ncT_CONCAT_sp16k_2/test.txt.encoded > nmt/models/multi30kT_ncT_CONCAT_sp16k_2/test.txt.2

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_ncT_CONCAT_sp16k_3/model.ep21.iter70581.bleu38.4.npz nmt/models/multi30kT_ncT_CONCAT_sp16k_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_ncT_CONCAT_sp16k_3/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_ncT_CONCAT_sp16k_3/test.txt.encoded > nmt/models/multi30kT_ncT_CONCAT_sp16k_3/test.txt.3

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_ncT_CONCAT_sp16k_1/model.ep16.iter53776.bleu38.58.npz,nmt/models/multi30kT_ncT_CONCAT_sp16k_2/model.ep15.iter50415.bleu39.23.npz,nmt/models/multi30kT_ncT_CONCAT_sp16k_3/model.ep21.iter70581.bleu38.4.npz nmt/models/multi30kT_ncT_CONCAT_sp16k_1/model.json,nmt/models/multi30kT_ncT_CONCAT_sp16k_2/model.json,nmt/models/multi30kT_ncT_CONCAT_sp16k_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30k_nc_ensemble_test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30k_nc_ensemble_test.txt.encoded > nmt/models/multi30k_nc_ensemble_test.txt

# Sub-word CONCAT Multitask (Multi30K)

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/model.ep14.iter47054.bleu38.39.npz nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/test.txt.encoded > nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/test.txt.1

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/model.ep16.iter53776.bleu38.59.npz nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/test.txt.encoded 1

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/test.txt.encoded > nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/test.txt.2

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/model.ep14.iter47054.bleu39.03.npz nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/test.txt.encoded > nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/test.txt.3

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/model.ep14.iter47054.bleu38.39.npz,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/model.ep16.iter53776.bleu38.59.npz,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/model.ep14.iter47054.bleu39.03.npz nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/model.json,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/model.json,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_multi30kI-InceptionV3_ncT_ensemble_test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_multi30kI-InceptionV3_ncT_ensemble_test.txt.encoded > nmt/models/multi30kT_multi30kI-InceptionV3_ncT_ensemble_test.txt

# Sub-word CONCAT Multitask (COCO)

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_1/model.ep16.iter53776.bleu38.9.npz nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_1/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_1/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_1/test.txt.encoded > nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_1/test.txt

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_2/model.ep22.iter73942.bleu38.82.npz nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_2/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_2/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_2/test.txt.encoded > nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_2/test.txt

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.ep16.iter53776.bleu39.29.npz nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/test.txt.encoded > nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/test.txt

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_1/model.ep16.iter53776.bleu38.9.npz,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_2/model.ep22.iter73942.bleu38.82.npz,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.ep16.iter53776.bleu39.29.npz nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.json,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.json,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_COCOI-InceptionV3_ncT_ensemble_test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_COCOI-InceptionV3_ncT_ensemble_test.txt.encoded > nmt/models/multi30kT_COCOI-InceptionV3_ncT_ensemble_test.txt

./test_decoding.sh nmt/data/sentencepiece16k/wmt_task1/test.en.norm.tok.lower.sp nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_1/model.ep16.iter53776.bleu38.9.npz,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_2/model.ep22.iter73942.bleu38.82.npz,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.ep16.iter53776.bleu39.29.npz,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/model.ep14.iter47054.bleu38.39.npz,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/model.ep16.iter53776.bleu38.59.npz,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/model.ep14.iter47054.bleu39.03.npz nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.json,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.json,nmt/models/multi30kT_COCOI-InceptionV3_ncT_CONCAT_16ksp_3/model.json,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_1/model.json,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_2/model.json,nmt/models/multi30kT_multi30kI-InceptionV3_ncT_CONCAT_16ksp_3/model.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/data/sentencepiece16k/joint.all.norm.tok.lower.sp.json nmt/models/multi30kT_COCOI_and_Multi30kI-InceptionV3_ncT_ensemble_test.txt.encoded 12

/usr/local/bin/spm_decode --model nmt/data/sentencepiece16k/joint.model < nmt/models/multi30kT_COCOI_and_Multi30kI-InceptionV3_ncT_ensemble_test.txt.encoded > nmt/models/multi30kT_COCOI_and_Multi30kI-InceptionV3_ncT_ensemble_test.txt

