import torch
from torchvision import datasets, transforms
from .sampling import *

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir
    if args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=transform_train)

        train_dataset, val_dataset = stratified_split(train_dataset)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = cifar_iid(train_dataset, args.num_users)
        else:
            user_groups_train = cifar_noniid(train_dataset, args.num_users, args.Nc)

    elif args.dataset == 'shakespears':
        print("ancora da finire")

    return train_dataset, val_dataset, test_dataset, user_groups_train


def get_user_input():
    print("Modify Parameters")

    # Modify iid parameter
    while True:
        iid = input("Enter IID parameter (1 for IID, 0 for non-IID): ")
        if iid in ['1', '0']:
            iid = int(iid)
            break
        else:
            print("Invalid input. Please enter 1 or 0.")

    # Modify participation parameter
    while True:
        participation = input("Enter Participation parameter (1 for Uniform, 0 for Skewed): ")
        if participation in ['1', '0']:
            participation = int(participation)
            break
        else:
            print("Invalid input. Please enter 1 or 0.")

    # If non-IID, ask for Nc parameter
    Nc = 50  # Default value
    if iid == 0:
        while True:
            Nc = input("Enter Nc parameter (1, 5, 10, 50): ")
            if Nc in ['1', '5', '10', '50']:
                Nc = int(Nc)
                break
            else:
                print("Invalid input. Please enter 1, 5, 10, or 50.")

    # Initialize J parameter
    J = 4

    # If non-IID, ask for J parameter
    if iid == 0:
        while True:
            J = input("Enter J parameter (4, 8, 16): ")
            if J in ['4', '8', '16']:
                J = int(J)
                break
            else:
                print("Invalid input. Please enter 4, 8, or 16.")

    return iid, participation, Nc, J


class DatasetSynthetic:
    def __init__(self, alpha, beta, theta, iid_sol, iid_data, n_dim, n_clnt, n_cls, avg_data, name_prefix):
        self.dataset = 'synt'
        self.name  = name_prefix + '_'
        self.name += '%d_%d_%d_%d_%f_%f_%f_%s_%s' %(n_dim, n_clnt, n_cls, avg_data,
                alpha, beta, theta, iid_sol, iid_data)

        data_path = 'Data'
        if (not os.path.exists('%s/%s/' %(data_path, self.name))):
            # Generate data
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data,
                                        alpha=alpha, beta=beta, theta=theta,
                                        iid_sol=iid_sol, iid_dat=iid_data)
            os.mkdir('%s/%s/' %(data_path, self.name))
            np.save('%s/%s/data_x.npy' %(data_path, self.name), data_x)
            np.save('%s/%s/data_y.npy' %(data_path, self.name), data_y)
        else:
            # Load data
            print('Load')
            data_x = np.load('%s/%s/data_x.npy' %(data_path, self.name), allow_pickle=True)
            data_y = np.load('%s/%s/data_y.npy' %(data_path, self.name), allow_pickle=True)

        for clnt in range(n_clnt):
            print(', '.join(['%.4f' %np.mean(data_y[clnt]==t) for t in range(n_cls)]))

        self.clnt_x = data_x
        self.clnt_y = data_y

        self.tst_x = np.concatenate(self.clnt_x, axis=0)
        self.tst_y = np.concatenate(self.clnt_y, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)

# Original prepration is from LEAF paper...
# This loads Shakespeare dataset only.
# data_path/train and data_path/test are assumed to be processed
# To make the dataset smaller,
# We take 2000 datapoints for each client in the train_set


class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Ignore groups information, combine test cases for different clients into one test data
        # Change structure to DatasetObject structure

        self.users = users

        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))

        tst_data_count = 0

        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[clnt]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start+crop_amount]

        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['y'])[start:start+curr_amount]

            tst_data_count += curr_amount

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        # Convert characters to numbers

        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)

        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)

        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))


        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))

            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)


        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))

        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Change structure to DatasetObject structure

        self.users = users

        tst_data_count_per_clnt = (crop_amount//tst_ratio)
        # Group clients that have at least crop_amount datapoints
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount
                and len(np.asarray(test_data[users[clnt]]['y'])) > tst_data_count_per_clnt):
                arr.append(clnt)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]

        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[idx]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:start+crop_amount]

        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):

            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
            tst_data_count += curr_amount

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        # Convert characters to numbers

        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)

        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)

        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))


        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))

            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)


        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))

        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'shakespeare':

            self.X_data = data_x
            self.y_data = data_y

            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx]
            return x, y


__all__ = ["get_dataset", "get_user_input", "DatasetSynthetic", "ShakespeareObjectCrop", "ShakespeareObjectCrop_noniid", "Dataset"]
