FROM ufoym/deepo:py36-cu90

CMD echo "Hello from inside the container"

CMD git clone https://github.com/micheledaddetta1/CycleNLPGAN.git
CMD cd CycleNLPGAN
CMD pip install -r requirements.txt
CMD pip install -U sentence-transformers
CMD pip install -e .
CMD pip install torch torchvision

CMD python ./train.py --dataroot data --on_colab True --batch_size 32 --task translation --name translation --checkpoints_dir ./checkpoints --freeze_GB_encoder True --eval_freq 4096
