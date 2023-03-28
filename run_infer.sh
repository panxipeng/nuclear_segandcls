python run_infer.py \
--gpu='0' \
--nr_types=5 \
--type_info_path=type_info.json \
--batch_size=1 \
--model_mode=fast \
--model_path=../model_weights/monusac_checkpoint.tar \ \
--nr_inference_workers=1 \
--nr_post_proc_workers=1 \
tile \
--input_dir=dataset/sample_images/imgs/ \
--output_dir=dataset/sample_images/pred/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath \
--save_raw_map
