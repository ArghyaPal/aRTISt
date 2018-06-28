from subprocess import check_output, call
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model_base_path = '/home/joseph/workspace/aRTISt/output/flowers_2018_06_22_13_37_37/model/'
eval_script_path = '/home/joseph/workspace/StackGAN-inception-model/'
artist_script_path = '/home/joseph/workspace/aRTISt/'

models = ['netG_'+str(i) for i in range(50000, 99000, 500)]

print len(models)

for model in models:
    print 'Processing: ', model

    # Updating the config file
    with open('./config/eval_flowers.yml', 'rt') as fin:
        with open('./temp.yml', 'wt') as fout:
            for line in fin:
                fout.write(line.replace('placeholder', model_base_path + model + '.pth'))
            fout.flush()

    # Calling main.py
    check_output(['python', './main.py', '--cfg', './temp.yml'])

    # Calling evaluation
    call(['cd', eval_script_path], shell=True)
    out = check_output(['python', eval_script_path + 'inception_score.py', 'image_folder', model_base_path + 'stage/'])
    call(['cd', artist_script_path], shell=True)

    with open('./output.txt', 'a') as fout:
        fout.write(model + '\n\n' + out + '\n++++++==========++++++\n')

    print 'Clearing stage.'
    call(['rm -rf', model_base_path+'/stage'], shell=True)
