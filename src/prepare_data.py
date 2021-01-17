#global imports
import os, random
#local imports


def fetch_data(random_authors=3, number_train_forms=2, number_test_forms=1, mode='train', debug = False):
    '''
        + This function is responsible for fetching random images for the system
        to be tuned.
        + Inputs: 
            random_authors : # of authors to be fetched.
            number_train_forms: # of train forms to get from them.
            number_test_forms: # of ...
            mode: if `train` .. you may tune the above parameters, if `test` they will be adjsuted to (3,2,1)
    '''
    if mode == 'train':
        data_path = 'data/' if debug else '../data/'

        datas = os.listdir(data_path)
        train_paths = []
        test_paths = []
        
        for _ in range (random_authors):
            user_i_train = []
            user_i_test = []
            rand_auth = random.choice(datas)
            while len(os.listdir(data_path+rand_auth)) < 3 or \
                    len(os.listdir(data_path+rand_auth)) < number_train_forms+number_test_forms:
                rand_auth = random.choice(datas)

            forms_paths = os.listdir(data_path+rand_auth)
            i = 0
            for form in forms_paths:
                if i < number_train_forms: 
                    user_i_train.append(data_path+rand_auth+'/'+form)

                elif number_test_forms+ number_train_forms > i:
                    user_i_test.append(data_path+rand_auth+'/'+form)
                else: break

                i += 1
            train_paths.append(user_i_train)
            test_paths.append(user_i_test)
        return train_paths, test_paths
