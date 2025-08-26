pip install -r requirements.txt


echo "Start enhancement..."

python code/python/test.py \
  --video data/dataset/s1_processed/videos/test/bbal8p.mpg \
  --ckpt  data/model/best_model.pth \
  --outdir runs/enhance_demo \
  --snr_db 8 \
  --recon mask \
  --smooth_t 3 \
  --hf_gain_db 5 \
  --hf_corner_hz 2500 \
  --hf_slope 1.2 \
  --audio_bitrate 256k \
  --container mp4 \
  --gain_db 25 \
  --lf_cut_db 4 --lf_corner_hz 160 --lf_slope 1.2 \
  --presence_gain_db 3 --presence_hz 3500 --presence_bw_hz 2000 \
  --hf_gain_db 5 --hf_corner_hz 3000 --hf_slope 1.1

echo "Done!\n";

echo "Extracting wave...";

python code/python/extract_wave.py \
  --noisy runs/enhance_demo/audio_noisy.wav \
  --enhanced runs/enhance_demo/audio_enhanced.wav \
  --clean runs/enhance_demo/clean_original.wav \
  --outdir spec_figs