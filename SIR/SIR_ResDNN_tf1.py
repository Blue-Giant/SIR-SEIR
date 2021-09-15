"""
@author: LXA
Benchmark Code of SIR model on tensorflow-1.14
2021-09-05
"""
import os
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import DNN_base
import DNN_tools
import DNN_data
import plotData
import saveData
import DNN_LogPrint

# The model of SIR is following
# dS/dt = -beta*S*I
# dI/dt = bets*S*I-gamma*I
# dR/dt = gamma*I


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model for SIR-parameters: %s\n' % str(R_dic['model2SIR_paras']), log_fileout)
    DNN_tools.log_string('activate function for SIR-parameters : %s\n' % str(R_dic['act2paras2SIR']), log_fileout)
    DNN_tools.log_string('Hidden layers for SIR-parameters: %s\n' % str(R_dic['hidden_list']), log_fileout)
    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']),
        log_fileout)

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']),
                         log_fileout)

    DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


class SIR_model(object):
    def __init__(self, in_dim=1, out_dim=1, model2DNN='DNN', hidden_layer=None, actIn_name='tanh', act_name='tanh',
                 actOut_name='linear', opt2regular_WB='L2'):
        super(SIR_model, self).__init__()
        self.DNN2beta = DNN_Class_base.Dense_Net(
            indim=in_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=model2DNN, actName2in=actIn_name,
            actName=act_name, actName2out=actOut_name, scope2W='Wbeta', scope2B='Bbeta')
        self.DNN2gamma = DNN_Class_base.Dense_Net(
            indim=in_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=model2DNN, actName2in=actIn_name,
            actName=act_name, actName2out=actOut_name, scope2W='Wgamma', scope2B='Bgamma')
        self.opt2regular_WB = opt2regular_WB

    def Solve_SIR(self, S_init=None, I_init=None, R_init=None, iter2SIR=None, t_data=None):
        """
        Args:
            S_init: The initial value for S, it is a positive real number
            I_init: The initial value for I, it is a positive real number
            R_init: The initial value for R, it is a positive real number
            iter2SIR: The iteration steps for solving SIR equations by means of Eular method or Runge-Kutta method
            t_data: an array of day, dim is [N, 1], the first element corresponds Init
        return:
            S_out: an array, it contains the S_init, dim is [N, 1]
            I_out: an array, it contains the I_init, dim is [N, 1]
            R_out: an array, it contains the R_init, dim is [N, 1]
        """
        bates = self.DNN2beta(t_data)      # [N,1]
        gammas = self.DNN2gamma(t_data)    # [N,1]
        S_out, I_out, R_out =[], [], []
        S_out.append(S_init)
        I_out.append(I_init)
        R_out.append(R_init)
        S_pre = S_init
        I_pre = I_init
        R_pre = R_init
        if iter2SIR is None:
            iter2SIR = int(len(t_data))

        for it in range(iter2SIR):
            beta = bates[it]
            gamma = gammas[it]
            S = S_pre - beta*S_pre*I_pre
            I = I_pre + beta*S_pre*I_pre - gamma*I_pre
            R = R_pre + gamma*I_pre
            S_out.append(S)
            I_out.append(I)
            R_out.append(R)
            S_pre = S
            I_pre = I
            R_pre = R
        return S_out, I_out, R_out

    def get_sumWB2DNNmodels(self):
        regular_WB2beta = self.DNN2beta.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        regular_WB2gamma = self.DNN2gamma.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        regular_WB = regular_WB2beta + regular_WB2gamma
        return regular_WB

    def Evaluate_SIR(self, S_init=None, I_init=None, R_init=None, iter2SIR=None, t_data=None):
        """
        Args:
            S_init: The initial value for S, it is a positive real number
            I_init: The initial value for I, it is a positive real number
            R_init: The initial value for R, it is a positive real number
            iter2SIR: The iteration steps for solving SIR equations by means of Eular method or Runge-Kutta method
            t_data: an array of day, dim is [N, 1], the first element corresponds Init
        return:
            S_out: an array, it contains the S_init, dim is [N, 1]
            I_out: an array, it contains the I_init, dim is [N, 1]
            R_out: an array, it contains the R_init, dim is [N, 1]
        """
        bates = self.DNN2beta(t_data)
        gammas = self.DNN2gamma(t_data)
        S_out, I_out, R_out = [], [], []
        S_out.append(S_init)
        I_out.append(I_init)
        R_out.append(R_init)
        S_pre = S_init
        I_pre = I_init
        R_pre = R_init
        if iter2SIR is None:
            iter2SIR = int(len(t_data))

        for it in range(iter2SIR):
            beta = bates[it]
            gamma = gammas[it]
            S = S_pre - beta*S_pre*I_pre
            I = I_pre + beta*S_pre*I_pre - gamma*I_pre
            R = R_pre + gamma*I_pre
            S_out.append(S)
            I_out.append(I)
            R_out.append(R)
            S_pre = S
            I_pre = I
            R_pre = R
        return S_out, I_out, R_out


def solve_SIR2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    log2trianSolus = open(os.path.join(log_out_path, 'train_Solus.txt'), 'w')      # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus = open(os.path.join(log_out_path, 'test_Solus.txt'), 'w')        # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus2 = open(os.path.join(log_out_path, 'test_Solus_temp.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    log2testParas = open(os.path.join(log_out_path, 'test_Paras.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    trainSet_szie = R['size2train']
    batchSize_train = R['batch_size2train']
    batchSize_test = R['batch_size2test']
    pt_penalty_init = R['init_penalty2predict_true']   # Regularization parameter for difference of predict and true
    penalty2wb = R['regular_weight']                   # Regularization parameter for weights
    lr_decay = R['lr_decay']
    tmp_lr = R['init_learning_rate']

    actFunc_name = R['act2paras2SIR']
    actOutFunc_name = R['actOut2paras2SIR']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            S_init2train = tf.placeholder(tf.float32, name='S_init2train', shape=[])
            I_init2train = tf.placeholder(tf.float32, name='I_init2train', shape=[])
            R_init2train = tf.placeholder(tf.float32, name='R_init2train', shape=[])
            T_batch2train = tf.placeholder(tf.float32, name='T_batch2train', shape=[batchSize_train, input_dim])

            S_true2train = tf.placeholder(tf.float32, name='S_true2train', shape=[batchSize_train, out_dim])
            I_true2train = tf.placeholder(tf.float32, name='I_true2train', shape=[batchSize_train, out_dim])
            R_true2train = tf.placeholder(tf.float32, name='R_true2train', shape=[batchSize_train, out_dim])
            N_true2train = tf.placeholder(tf.float32, name='N_true2train', shape=[batchSize_train, out_dim])

            S_init2test = tf.placeholder(tf.float32, name='S_init2test', shape=[])
            I_init2test = tf.placeholder(tf.float32, name='I_init2test', shape=[])
            R_init2test = tf.placeholder(tf.float32, name='R_init2test', shape=[])
            T_batch2test = tf.placeholder(tf.float32, name='T_batch2test', shape=[batchSize_test, input_dim])

            pt_penalty = tf.placeholder_with_default(input=1e1, shape=[], name='penalty2predict_true')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')

            S_NN2train, I_NN2train, R_NN2train = Res_DNN2SIR.Solve_SIR(
                S_init=S_init2train, I_init=I_init2train, R_init=R_init2train, t_data=T_batch2train)
            loss2S = tf.reduce_mean(tf.square(S_NN2train - S_true2train))
            loss2I = tf.reduce_mean(tf.square(I_NN2train - I_true2train))
            loss2R = tf.reduce_mean(tf.square(R_NN2train - R_true2train))
            N_NN = S_NN2train + I_NN2train + R_NN2train
            loss2N = tf.reduce_mean(tf.square(N_NN - N_true2train))
            PWB = penalty2wb * Res_DNN2SIR.get_sumWB2DNNmodels()
            loss = loss2S + loss2R + loss2I + loss2N + PWB

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_loss = my_optimizer.minimize(loss, global_step=global_steps)

            S_NN2test, I_NN2test, R_NN2test = Res_DNN2SIR.Evaluate_SIR(
                S_init=S_init2test, I_init=I_init2test, R_init=R_init2test, t_data=T_batch2test)

    # 载入数据集
    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    filename = 'data2csv/Korea_data.csv'
    date, data = DNN_data.load_csvData(filename)

    # 将数据集分为训练集和测试集
    assert(trainSet_szie + batchSize_test <= len(data))
    train_date, train_data2i, test_date, test_data2i = DNN_data.split_csvData2train_test(
        date, data, size2train=trainSet_szie, normalFactor=R['scale_population'])

    # 归一化数据集的模式
    if R['scale_population'] == 1:
        nbatch2train = np.ones(batchSize_train, dtype=np.float32)*float(R['total_population'])
    elif (R['total_population'] != R['scale_population']) and R['scale_population'] != 1:
        nbatch2train = np.ones(batchSize_train, dtype=np.float32) * (float(R['total_population'])/float(R['scale_population']))
    elif (R['total_population'] == R['scale_population']) and R['scale_population'] != 1:
        nbatch2train = np.ones(batchSize_train, dtype=np.float32)

    # 从测试集中选出一定的数据作为测试批次。对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)
    i_obs_test = DNN_data.sample_testData_serially(test_data2i, batchSize_test, normalFactor=1.0)
    print('The test data for i:\n', str(np.transpose(i_obs_test)))
    print('\n')
    DNN_tools.log_string('The test data about i:\n%s\n' % str(np.transpose(i_obs_test)), log_fileout)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_n_all, loss_all = [], [], [], [], []
    test_epoch = []
    test_mse2I_all, test_rel2I_all = [], []

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i_epoch in range(R['max_epoch'] + 1):
            # 从训练集中选出数据作为训练批次
            t_batch, i_obs = DNN_data.randSample_Normalize_existData(
                train_date, train_data2i, batchsize=batchSize_train, normalFactor=1.0, sampling_opt=R['opt2sample'])
            n_obs = nbatch2train.reshape(batchSize_train, 1)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_pt = 10 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_pt = 50 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_pt = 100 * pt_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_pt = 200 * pt_penalty_init
                else:
                    temp_penalty_pt = 500 * pt_penalty_init
            else:
                temp_penalty_pt = pt_penalty_init

            loss_s_all.append(loss2S.numpy())
            loss_i_all.append(loss2I.numpy())
            loss_r_all.append(loss2R.numpy())
            loss_n_all.append(loss2N.numpy())
            loss_all.append(loss.numpy())

            if i_epoch % 1000 == 0:
                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的训练结果
                DNN_LogPrint.print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_pt, loss_s, loss_i,
                                                 loss_r, loss_n, loss, log_out=log_fileout)

                s_nn2train, i_nn2train, r_nn2train = sess.run(
                    [SNN, INN, RNN], feed_dict={T_it: np.reshape(train_date, [-1, 1])})

                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的测试结果
                test_epoch.append(i_epoch / 1000)
                train_option = False
                s_nn2test, i_nn2test, r_nn2test, beta_test, gamma_test = sess.run(
                    [SNN, INN, RNN, beta, gamma], feed_dict={T_it: test_t_bach, train_opt: train_option})
                point_ERR2I = np.square(i_nn2test - i_obs_test)
                test_mse2I = np.mean(point_ERR2I)
                test_mse2I_all.append(test_mse2I)
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2I_all.append(test_rel2I)

                DNN_tools.print_and_log_test_one_epoch(test_mse2I, test_rel2I, log_out=log_fileout)
                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testSolus)
                DNN_tools.log_string('The test result for s:\n%s\n' % str(np.transpose(s_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for i:\n%s\n' % str(np.transpose(i_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for r:\n%s\n\n' % str(np.transpose(r_nn2test)), log2testSolus)

                # --------以下代码为输出训练过程中 S_NN_temp, I_NN_temp, R_NN_temp, in_beta, in_gamma 的测试结果-------------
                s_nn_temp2test, i_nn_temp2test, r_nn_temp2test, in_beta_test, in_gamma_test = sess.run(
                    [SNN_temp, INN_temp, RNN_temp, in_beta, in_gamma],
                    feed_dict={T_it: test_t_bach, train_opt: train_option})

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testSolus2)
                DNN_tools.log_string('The test result for s_temp:\n%s\n' % str(np.transpose(s_nn_temp2test)), log2testSolus2)
                DNN_tools.log_string('The test result for i_temp:\n%s\n' % str(np.transpose(i_nn_temp2test)), log2testSolus2)
                DNN_tools.log_string('The test result for r_temp:\n%s\n\n' % str(np.transpose(r_nn_temp2test)), log2testSolus2)

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testParas)
                DNN_tools.log_string('The test result for in_beta:\n%s\n' % str(np.transpose(in_beta_test)), log2testParas)
                DNN_tools.log_string('The test result for in_gamma:\n%s\n' % str(np.transpose(in_gamma_test)), log2testParas)

        DNN_tools.log_string('The train result for S:\n%s\n' % str(np.transpose(s_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for I:\n%s\n' % str(np.transpose(i_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for R:\n%s\n\n' % str(np.transpose(r_nn2train)), log2trianSolus)

        saveData.true_value2convid(train_data2i, name2Array='itrue2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(s_nn2train, name2solus='s2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(i_nn2train, name2solus='i2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(r_nn2train, name2solus='r2train', outPath=R['FolderName'])

        saveData.save_SIR_trainLoss2mat_Covid(loss_s_all, loss_i_all, loss_r_all, loss_n_all, actName=act_func2SIR,
                                              outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_n_all, lossType='loss2n', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        saveData.true_value2convid(i_obs_test, name2Array='i_true2test', outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse2I_all, test_rel2I_all, actName='Infected', outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse2I_all, test_rel2I_all, test_epoch, actName='Infected', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        saveData.save_SIR_testSolus2mat_Covid(s_nn2test, i_nn2test, r_nn2test, name2solus1='snn2test',
                                              name2solus2='inn2test', name2solus3='rnn2test', outPath=R['FolderName'])
        saveData.save_SIR_testParas2mat_Covid(beta_test, gamma_test, name2para1='beta2test', name2para2='gamma2test',
                                              outPath=R['FolderName'])

        plotData.plot_testSolu2convid(i_obs_test, name2solu='i_true', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(s_nn2test, name2solu='s_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(i_nn2test, name2solu='i_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(r_nn2test, name2solu='r_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])

        plotData.plot_testSolus2convid(i_obs_test, i_nn2test, name2solu1='i_true', name2solu2='i_test',
                                       coord_points2test=test_t_bach, seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plot_testSolu2convid(beta_test, name2solu='beta_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(gamma_test, name2solu='gamma_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SIR2covid'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ----------------------------------------- Convid 设置 ---------------------------------
    R['eqs_name'] = 'SIR'
    R['input_dim'] = 1                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['total_population'] = 9776000

    R['scale_population'] = 9776000
    # R['scale_population'] = 100000
    # R['scale_population'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['size2train'] = 70                    # 训练集的大小
    R['batch_size2train'] = 20              # 训练数据的批大小
    R['batch_size2test'] = 10               # 训练数据的批大小
    # R['opt2sample'] = 'random_sample'     # 训练集的选取方式--随机采样
    R['opt2sample'] = 'rand_sample_sort'    # 训练集的选取方式--随机采样后按时间排序

    R['init_penalty2predict_true'] = 50   # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1       # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        # R['init_penalty2predict_true'] = 1000
        # R['init_penalty2predict_true'] = 100
        # R['init_penalty2predict_true'] = 50
        # R['init_penalty2predict_true'] = 20
        R['init_penalty2predict_true'] = 1

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight'] = 0.000             # Regularization parameter for weights

    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'        # The model of regular weights and biases
    # R['regular_weight'] = 0.001           # Regularization parameter for weights
    # R['regular_weight'] = 0.0005          # Regularization parameter for weights
    # R['regular_weight'] = 0.0001            # Regularization parameter for weights
    R['regular_weight'] = 0.00005          # Regularization parameter for weights
    # R['regular_weight'] = 0.00001        # Regularization parameter for weights

    R['optimizer_name'] = 'Adam'           # 优化器
    R['loss_function'] = 'L2_loss'
    R['scale_up'] = 1
    R['scale_factor'] = 100
    # R['loss_function'] = 'lncosh_loss'
    # R['train_model'] = 'train_group'
    R['train_model'] = 'train_union_loss'

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-3           # 学习率
        R['lr_decay'] = 1e-4                # 学习率 decay
        # R['init_learning_rate'] = 2e-4    # 学习率
        # R['lr_decay'] = 5e-5              # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        # R['init_learning_rate'] = 1e-3    # 学习率
        # R['lr_decay'] = 1e-4              # 学习率 decay
        # R['init_learning_rate'] = 2e-4    # 学习率
        # R['lr_decay'] = 1e-4              # 学习率 decay
        R['init_learning_rate'] = 1e-4      # 学习率
        R['lr_decay'] = 5e-5                # 学习率 decay
    else:
        R['init_learning_rate'] = 5e-5      # 学习率
        R['lr_decay'] = 1e-5                # 学习率 decay

    # 网络模型的选择
    R['model2SIR_paras'] = 'DNN'
    # R['model2SIR_paras'] = 'DNN_modify'
    # R['model2SIR_paras'] = 'DNN_scale'
    # R['model2SIR_paras'] = 'DNN_FourierBase'

    if R['model2SIR_paras'] == 'DNN_FourierBase':
        R['hidden_list'] = (35, 50, 30, 30, 20)         # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # R['hidden_list'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        R['hidden_list'] = (70, 50, 30, 30, 20)         # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # R['hidden_list'] = (80, 80, 60, 40, 40, 20)   # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # R['hidden_list'] = (100, 100, 80, 60, 60, 40)
        # R['hidden_list'] = (200, 100, 100, 80, 50, 50)


    # R['act2paras2SIR'] = 'relu'
    R['act2paras2SIR'] = 'tanh'
    # R['act2paras2SIR'] = 'sigmod'
    # R['act2paras2SIR'] = 'leaky_relu'
    # R['act2paras2SIR'] = 'srelu'
    # R['act2paras2SIR'] = 's2relu'
    # R['act2paras2SIR'] = 'elu'
    # R['act2paras2SIR'] = 'selu'
    # R['act2paras2SIR'] = 'phi'

    R['actOut2paras2SIR'] = 'relu'

    solve_SIR2COVID(R)
