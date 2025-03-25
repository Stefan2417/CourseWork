#!/bin/bash
#SBATCH --job-name=train_adapter_w2v_fixed            # Название задачи
#SBATCH --error=task-%j.err        # Файл для вывода ошибок
#SBATCH --output=task-%j.log       # Файл для вывода результатов
#SBATCH --time=200:00:00            # Максимальное время выполнения
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=16          # Требуемое кол-во CPU
#SBATCH --mail-user=stankuznetsov@edu.hse.ru    # Укажи�~Bе ва�~H email дл�~O о�~Bп�~@авки �~Cведомлений
#SBATCH --mail-type=END,FAIL               # Соб�~K�~Bи�~O, �~B�~@еб�~C�~N�~Iие �~Cведомлени�~O

python /home/stankuznetsov/CourseWork/train.py --config-name train_w2v_adapter_server writer.run_name="train_adapter_w2v_fixed" dataloader.workers=16 dataloader.batch_size=256  model.freeze_strategy='all'                       # Выполнение расчёта
#srun {options} {command}            # Выполнение MPI расчёта