# Illustrate
main.exe \ 
--encoder_model_path {your_encoder_model_path} # Encoder model path , required
--decoder_model_path {your_decoder_model_path} # Decoder model path , required
--image_path {image_path} # Image path , required
--save_dir {save_path} # Save path , required
--use_demo true # Whether run SAM with graphical interface demo , default true
--use_boxinfo true # Whether use box prompt information , default true
--use_singlemask false # Whether use singlemask model , default false
--threshold 0.9 # Set threshold , default 0.9

# Sample
## vit-b
Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_b\\sam_vit_b_01ec64_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_b\\sam_vit_b_01ec64_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\input\\1_1-2.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_boxinfo true --use_demo true

Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_b\\sam_vit_b_01ec64_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_b\\sam_vit_b_01ec64_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\truck.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo false --use_boxinfo true

Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_b\\sam_vit_b_01ec64_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_b\\sam_vit_b_01ec64_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\dog.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true

# vit-l dog.jpg
Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\dog.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true --use_boxinfo true

Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\dog.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true

Release\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\dog.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true

## vit-l truck.jpg
Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\truck.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true --use_boxinfo true

Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\truck.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output

Release\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\assets\\truck.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true --use_boxinfo true

## vit-l 1_1-2.jpg
Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\input\\4_2-2.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true --use_boxinfo true

Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\input\\1_1-2.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true --use_boxinfo true

Debug\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\input\\1_1-2.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true --use_boxinfo true

Release\main.exe --encoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\encoder\\vit_l\\sam_vit_l_0b3195_encoder.onnx --decoder_model_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\models\\decoder\\vit_l\\sam_vit_l_0b3195_decoder.onnx --image_path D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\input\\1_1-2.jpg --save_dir D:\\OroChiLab\\SegmentAnything-OnnxRunner\\data\\output --use_demo true --use_boxinfo true
