import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle
import torch
from sklearn.metrics import classification_report
from model.SBAG import SBAG, SBAG_weibo



def load_dataset(task):
    print("task: ", task)

    A_us, A_uu = pickle.load(open("dataset/"+task+"/relations.pkl", 'rb'))
    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, word_embeddings = pickle.load(open("dataset/"+task+"/train.pkl", 'rb'))
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev = pickle.load(open("dataset/"+task+"/dev.pkl", 'rb'))
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = pickle.load(open("dataset/"+task+"/test.pkl", 'rb'))
   
    config['maxlen'] = len(X_train_source_wid[0])
    if task == 'twitter15':
        config['n_heads'] = 7
    elif task == 'twitter16':
        config['n_heads'] = 7
    else:
        config['n_heads'] = 12
        config['batch_size'] = 128
        config['num_classes'] = 2
        config['target_names'] = ['NR', 'FR']

    config['embedding_weights'] = word_embeddings
    config['A_us'] = A_us
    config['A_uu'] = A_uu
    return X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
           X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
           X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test


def train_and_test(model, task):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'checkpoint/weights.best.' + task + "." + model_suffix

    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = load_dataset(task)

    nn = model(config)
    nn.fit(X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,
           X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev)

    print("================================")
    nn.load_state_dict(torch.load(config['save_path']))
    y_pred = nn.predict(X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, task, test_flag = True)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))


config = {
    'lr':1e-3,
    'reg':1e-6,
    'embeding_size': 100,
    'batch_size':16,
    'nb_filters':100,
    'kernel_sizes':[3, 4, 5],
    'dropout':0.5,
    'epochs':20,
    'num_classes':4,
    'target_names':['NR', 'FR', 'UR', 'TR']
}


if __name__ == '__main__':
    #task = 'twitter15'
    #task = 'twitter16'
    task = 'weibo'
    if task == 'weibo':
        model = SBAG_weibo
        train_and_test(model, task)
    else:        
        model = SBAG
        train_and_test(model, task)


