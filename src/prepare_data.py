#global imports
import os, random, shutil
#local imports



def fetch_deliver(test_folder):
    
    train_paths = []
    test_paths = []
    test_cases = os.listdir('data/'+test_folder)
    test_cases = sorted(test_cases)
    for test_case in test_cases:
        if test_case == 'test.png':
            test_paths.append('data/'+test_folder+'/test.png') 
            # print ('data/'+test_folder+'/test.png')   
        else:
            forms = os.listdir('data/'+test_folder+'/'+test_case)
            forms = sorted(forms)
            for form in forms:
                train_paths.append('data/'+test_folder+'/'+test_case+'/'+form)
                # print ('data/'+test_folder+'/'+test_case+'/'+form)

    return train_paths, test_paths

def read_brute(set_id,form_id):
    data_path = 'data_tune/' 

    datas = os.listdir(data_path)
    train_paths = []
    test_paths = []
    chosen_one = datas[set_id]
    rand_author_test_sample = random.randint(1,3)
    authors_picked = []
    for author_i in range (1,4):
        user_i_train = []
        user_i_test = []

        rand_auth = random.choice(datas)
        while len(os.listdir(data_path+rand_auth)) < 3 or \
                rand_auth in authors_picked or \
                rand_auth == chosen_one:

            rand_auth = random.choice(datas)

        if author_i == rand_author_test_sample:
            authors_picked.append(chosen_one)            
            forms_paths = os.listdir(data_path+chosen_one)
            forms_chosen = forms_paths[form_id]
            user_i_test.append(data_path + chosen_one+'/'+forms_chosen)
            forms_paths.remove(forms_chosen)
        else:
            authors_picked.append(rand_auth)            
            forms_paths = os.listdir(data_path+rand_auth)


        i = 0
        
        for form in forms_paths:
            if author_i == rand_author_test_sample:
                if i < 2: 
                    user_i_train.append(data_path+chosen_one+'/'+form)
                else: break
                i += 1
            else:
                if i < 2: 
                    user_i_train.append(data_path+rand_auth+'/'+form)
                else: break
                i += 1
        train_paths.append(user_i_train)
        test_paths.append(user_i_test)
    return train_paths, test_paths


def fetch_data(random_authors=3, number_train_forms=2, number_test_forms=1, _mode='test',set_id=None,form_id=None):
    '''
        + This function is responsible for fetching random images for the system
        to be tuned.
        + Inputs: 
            random_authors : # of authors to be fetched.
            number_train_forms: # of train forms to get from them.
            number_test_forms: # of ...
            _mode: if `train` .. you may tune the above parameters, if `test` they will be adjsuted to (3,2,1)
    '''
    if set_id is not None:
        return read_brute(set_id,form_id)

    if _mode == 'train':
        data_path = 'data_tune/'

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
        data_path = 'data_tune/' 

        datas = os.listdir(data_path)
        train_paths = []
        test_paths = []
        
        rand_author_test_sample = random.randint(0,2)
        authors_picked = []
        for author_i in range (3):
            user_i_train = []
            user_i_test = []

            rand_auth = random.choice(datas)
            while len(os.listdir(data_path+rand_auth)) < 3 or rand_auth in authors_picked:
                rand_auth = random.choice(datas)

            authors_picked.append(rand_auth)            

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

def move_extras():
    if not os.path.exists('data_ex'):
        os.mkdir('data_ex')
    data_folders = os.listdir('data_tune/')
    for data_folder in data_folders:
        folder_path = 'data_tune/'+data_folder
        new_path = 'data_ex/'+data_folder
        if len(os.listdir(folder_path)) < 3:
            #move it aways
            shutil.move(folder_path, new_path)



def data_stat_new(): #899
    count = 0
    datas = os.listdir('data_tune/')
    for data in datas:
        count += len(os.listdir('data_tune/'+data))
    print (count)
