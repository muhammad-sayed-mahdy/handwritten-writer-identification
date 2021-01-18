#global imports
import os, random
#local imports


def fetch_data(random_authors=3, number_train_forms=2, number_test_forms=1, _mode='train', debug = False):
    '''
        + This function is responsible for fetching random images for the system
        to be tuned.
        + Inputs: 
            random_authors : # of authors to be fetched.
            number_train_forms: # of train forms to get from them.
            number_test_forms: # of ...
            _mode: if `train` .. you may tune the above parameters, if `test` they will be adjsuted to (3,2,1)
    '''
    if _mode == 'train':
        data_path = 'data/'

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
        
    elif _mode == 'test':
        data_path = 'data/' 

        datas = os.listdir(data_path)
        train_paths = []
        test_paths = []
        
        rand_author_test_sample = random.randint(0,2)
        for author_i in range (3):
            user_i_train = []
            user_i_test = []
            rand_auth = random.choice(datas)
            while len(os.listdir(data_path+rand_auth)) < 3 or \
                    len(os.listdir(data_path+rand_auth)) < 2+1:
                rand_auth = random.choice(datas)

            forms_paths = os.listdir(data_path+rand_auth)
            i = 0
            for form in forms_paths:
                if i < 2: 
                    user_i_train.append(data_path+rand_auth+'/'+form)

                elif i == 2 and author_i == rand_author_test_sample:
                    user_i_test.append(data_path+rand_auth+'/'+form)
                else: break

                i += 1
            train_paths.append(user_i_train)
            test_paths.append(user_i_test)
        return train_paths, test_paths


def print_data_stat():
    data = os.listdir('../data/')
    sizes = []
    imp_authors = {}
    file_size = {}
    data = sorted(data)
    for x in data:
        files = os.listdir('../data/'+x)
        file_size[x] = len(files)
        if len(files) >= 3:
            imp_authors[x] = len(files)
        sizes.append(len(files))

    # print ('Dict of sizes', file_size)
    print ('List of sizes in order:',sizes)
    print ('Total Number of Authors:',len(file_size))
    print ('Unique Sizes:',set(sizes))
    print ('Imp authors with more than 2 files:',imp_authors)
    print ('Total number of important authors:',len(imp_authors))