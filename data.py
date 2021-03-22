import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py

import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import rand_float, rand_int
from utils import sample_control_RiceGrip, calc_shape_states_RiceGrip
from utils import calc_box_init_FluidShake, calc_shape_states_FluidShake


def collate_fn(data):
    return data[0]


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].size(1) / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(-1, n_rep * stat_dim)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].shape[1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data


def rotateByQuat(p, quat):
    R = np.zeros((3, 3))
    a, b, c, d = quat[3], quat[0], quat[1], quat[2]
    R[0, 0] = a**2 + b**2 - c**2 - d**2
    R[0, 1] = 2 * b * c - 2 * a * d
    R[0, 2] = 2 * b * d + 2 * a * c
    R[1, 0] = 2 * b * c + 2 * a * d
    R[1, 1] = a**2 - b**2 + c**2 - d**2
    R[1, 2] = 2 * c * d - 2 * a * b
    R[2, 0] = 2 * b * d - 2 * a * c
    R[2, 1] = 2 * c * d + 2 * a * b
    R[2, 2] = a**2 - b**2 - c**2 + d**2

    return np.dot(R, p)


def gen_PyFleX(info):

    env, root_num = info['env'], info['root_num']
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, n_instance = info['n_rollout'], info['n_instance']
    time_step, time_step_clip = info['time_step'], info['time_step_clip']
    shape_state_dim, dt = info['shape_state_dim'], info['dt']

    env_idx = info['env_idx'] # =11

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2**32)
    
    stats = [init_stat(3), init_stat(3)]

    import pyflex
    pyflex.init()

    for i in range(n_rollout):

        if i % 10 == 0:
            print("%d / %d" % (i, n_rollout))

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)
        
        # scene_params: [len(box) at dim x,len(box) at dim y,len(box) at dim z, num_hair per circle, num_circle]
        cap_size = [0.1,1.5]
        N_hairs = 1


        scene_params = np.array(cap_size)

        pyflex.set_scene(env_idx, scene_params, thread_idx)
        n_particles = pyflex.get_n_particles()
        n_shapes = 1
        N_particles_per_hair = int(n_particles/N_hairs)
        idx_begins = np.arange(N_hairs)*N_particles_per_hair
        idx_hairs = [[i,i+N_particles_per_hair-1] for i in idx_begins]

        positions = np.zeros((time_step, n_particles+ n_shapes, 3), dtype=np.float32)
        velocities = np.zeros((time_step, n_particles+ n_shapes, 3), dtype=np.float32)
   #     shape_position = np.zeros((time_step, n_shapes, 3), dtype=np.float32)
    #    shape_velocities = np.zeros((time_step, n_shapes, 3), dtype=np.float32)

        for j in range(time_step_clip):
         #   p_clip = pyflex.get_positions().reshape(-1, 4)[:, :3]
         #   shape_p_clip = pyflex.get_shape_states()[:3].reshape(-1,3)
            p_clip = np.concatenate([pyflex.get_positions().reshape(-1, 4)[:, :3],pyflex.get_shape_states()[:3].reshape(-1,3)],axis = 0)
            pyflex.step()

        for j in range(time_step):
            positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]
            for k in range(n_shapes):
                 positions[j, n_particles + k] = pyflex.get_shape_states()[:3]
        #    shape_position[j] = pyflex.get_shape_states()[:3].reshape(-1,3)
        #    shape_prevposition = pyflex.get_shape_states()[3:6].reshape(-1,3)
            if j == 0:
                velocities[j] = (positions[j] - p_clip) / dt
           #     shape_velocities[j] = (shape_position[j] - shape_p_clip)/dt
            else:
                velocities[j] = (positions[j] - positions[j - 1]) / dt
          #      shape_velocities[j] = (shape_position[j] - shape_position[j-1])/dt

            pyflex.step()
            data = [positions[j], velocities[j], idx_hairs]
            store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))
        
        # change dtype for more accurate stat calculation
        # only normalize positions and velocities
        datas = [positions.astype(np.float64), velocities.astype(np.float64)]

        for j in range(len(stats)): 
            # here j = 2, refers to positions and velocities
            stat = init_stat(stats[j].shape[0]) 
            # stat= np.zeros((3,3))
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1] # time_step*n_particles
            stats[j] = combine_stat(stats[j], stat)

    pyflex.clean()

    return stats



def visualize_neighbors(anchors, queries, idx, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    ax.scatter(anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c='r', s=80)
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    ax.set_aspect('equal')

    plt.show()


def find_relations_neighbor(positions, query_idx, anchor_idx, radius, order, var=False):
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions

    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender, np.ones(count_neighbors)], axis=1))

    return relations


def make_hierarchy(env, attr, positions, velocities, idx, st, ed, phases_dict, count_nodes, clusters, verbose=0, var=False):
    order = 2
    n_root_level = len(phases_dict["root_num"][idx])
    attr, relations, node_r_idx, node_s_idx, pstep = [attr], [], [], [], []

    relations_rev, node_r_idx_rev, node_s_idx_rev, pstep_rev = [], [], [], []

    pos = positions.data.cpu().numpy() if var else positions
    vel = velocities.data.cpu().numpy() if var else velocities

    for i in range(n_root_level):
        root_num = phases_dict["root_num"][idx][i]
        root_sib_radius = phases_dict["root_sib_radius"][idx][i]
        root_des_radius = phases_dict["root_des_radius"][idx][i]
        root_pstep = phases_dict["root_pstep"][idx][i]

        if verbose:
            print('root info', root_num, root_sib_radius, root_des_radius, root_pstep)


        ### clustring the nodes
        # st_time = time.time()
        # kmeans = MiniBatchKMeans(n_clusters=root_num, random_state=0).fit(pos[st:ed, 3:6])
        # print('Time on kmeans', time.time() - st_time)
        # clusters = kmeans.labels_
        # centers = kmeans.cluster_centers_


        ### relations between roots and desendants
        rels, rels_rev = [], []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(st, ed))
        node_r_idx_rev.append(node_s_idx[-1])
        node_s_idx_rev.append(node_r_idx[-1])
        pstep.append(1); pstep_rev.append(1)

        if verbose:
            centers = np.zeros((root_num, 3))
            for j in range(root_num):
                des = np.nonzero(clusters[i][0]==j)[0]
                center = np.mean(pos[st:ed][des, -3:], 0, keepdims=True)
                centers[j] = center[0]
                visualize_neighbors(pos[st:ed], center, 0, des)

        for j in range(root_num):
            desendants = np.nonzero(clusters[i][0]==j)[0]
            roots = np.ones(desendants.shape[0]) * j
            if verbose:
                print(roots, desendants)
            rels += [np.stack([roots, desendants, np.zeros(desendants.shape[0])], axis=1)]
            rels_rev += [np.stack([desendants, roots, np.zeros(desendants.shape[0])], axis=1)]
            if verbose:
                print(np.max(np.sqrt(np.sum(np.square(pos[st + desendants, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))
        relations_rev.append(np.concatenate(rels_rev, 0))


        ### relations between roots and roots
        # point_tree = spatial.cKDTree(centers)
        # neighbors = point_tree.query_ball_point(centers, root_sib_radius, p=order)

        '''
        for j in range(len(neighbors)):
            visualize_neighbors(centers, centers, j, neighbors[j])
        '''

        rels = []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(count_nodes, count_nodes + root_num))
        pstep.append(root_pstep)

        roots = np.repeat(np.arange(root_num), root_num)
        siblings = np.tile(np.arange(root_num), root_num)
        if verbose:
            print(roots, siblings)
        rels += [np.stack([roots, siblings, np.zeros(root_num * root_num)], axis=1)]
        if verbose:
            print(np.max(np.sqrt(np.sum(np.square(centers[siblings, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))


        ### add to attributes/positions/velocities
        positions = [positions]
        velocities = [velocities]
        attributes = []
        for j in range(root_num):
            ids = np.nonzero(clusters[i][0]==j)[0]
            if var:
                positions += [torch.mean(positions[0][st:ed, :][ids], 0, keepdim=True)]
                velocities += [torch.mean(velocities[0][st:ed, :][ids], 0, keepdim=True)]
            else:
                positions += [np.mean(positions[0][st:ed, :][ids], 0, keepdims=True)]
                velocities += [np.mean(velocities[0][st:ed, :][ids], 0, keepdims=True)]

            attributes += [np.mean(attr[0][st:ed, :][ids], 0, keepdims=True)]

        attributes = np.concatenate(attributes, 0)

        if env == 'BoxBath':
            attributes[:, 2 + i] = 1
        elif env == 'RiceGrip':
            attributes[:, 1 + i] = 1

        if verbose:
            print('Attr sum', np.sum(attributes, 0))

        attr += [attributes]
        if var:
            positions = torch.cat(positions, 0)
            velocities = torch.cat(velocities, 0)
        else:
            positions = np.concatenate(positions, 0)
            velocities = np.concatenate(velocities, 0)

        st = count_nodes
        ed = count_nodes + root_num
        count_nodes += root_num

        if verbose:
            print(st, ed, count_nodes, positions.shape, velocities.shape)

    attr = np.concatenate(attr, 0)
    if verbose:
        print("attr", attr.shape)

    relations += relations_rev[::-1]
    node_r_idx += node_r_idx_rev[::-1]
    node_s_idx += node_s_idx_rev[::-1]
    pstep += pstep_rev[::-1]

    return attr, positions, velocities, count_nodes, relations, node_r_idx, node_s_idx, pstep


def prepare_input(data, stat, args, phases_dict, verbose=0, var=False):
    '''
    for a single hair
    '''
    positions, velocities, hairs_idx = data
    n_shapes = 1
    hairs_idx_begin = [idx[0] for idx in hairs_idx]
    n_particles = positions.shape[0] - n_shapes
    R = 0.15
    
    ### object attributes
    #   dim 10: [rigid, fluid, root_0, root_1, gripper_0, gripper_1, mass_inv,
    #            clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep]
    #   here we only consider the hairs but not the gripper, attr_dim = 1, attr = 0 for hair, attr = 1 for shapes
    attr = np.zeros((n_particles+n_shapes, args.attr_dim))
    
    ### construct relations
    Rr_idxs = []        # relation receiver idx list
    Rs_idxs = []        # relation sender idx list
    Ras = []            # relation attributes list
    values = []         # relation value list (should be 1)
    node_r_idxs = []    # list of corresponding receiver node idx
    node_s_idxs = []    # list of corresponding sender node idx
    psteps = []         # propagation steps
    
    ##### add env specific graph components
    ### specify for shapes
    rels = []
    vals = []
    
    for i in range(n_shapes):
        attr[n_particles+i, 0] = 1
        dis = np.linalg.norm(positions[:n_particles,:2]-positions[n_particles+i,:2],axis = 1)
        if verbose:
            print ("dis:", dis)
        nodes_rel = np.nonzero(dis <= R)[0]
        # for relation between hair nodes and a gripper, we note it as 1
        gripper = np.ones(nodes_rel.shape[0], dtype=np.int) * (n_particles+i)
        rels += [np.stack([nodes_rel, gripper, np.ones(nodes_rel.shape[0])], axis=1)]
        if verbose:
            print ("dis val:", dis[nodes_rel])
      #  vals += [np.ones(nodes_rel.shape[0], dtype=np.int)]
        vals += [dis[nodes_rel]]
        
    
    ##### add relations between leaf particles
    ## here we only consider the relations in a hair: the relation between a node and the nodes nearby
    ## simple case for one hair, TEMPORARY 2 rels for one link
    nodes_p = np.arange(n_particles-1)
    val = np.linalg.norm(positions[1:n_particles]-positions[:n_particles-1],axis = 1)
    R1 = np.stack([nodes_p,nodes_p+1, np.zeros(n_particles-1)],axis = 1)
    R2 = np.stack([nodes_p+1,nodes_p, np.zeros(n_particles-1)],axis = 1)
    rels += [np.concatenate([R1,R2],axis = 0)]
    vals += [val,val]
    
    rels = np.concatenate(rels, 0)
    vals = np.concatenate(vals, 0)
    
  #  print (vals.shape)
 #   print (rels.shape)
    
    
    if rels.shape[0] > 0:
        if verbose:
            print("Relations neighbor", rels.shape)
        Rr_idxs.append(torch.LongTensor([rels[:, 0], np.arange(rels.shape[0])]))
        Rs_idxs.append(torch.LongTensor([rels[:, 1], np.arange(rels.shape[0])]))
        # Ra: relation attributes
    #    Ra = np.zeros((rels.shape[0], args.relation_dim))  
        Ra = rels[:,2].reshape([-1,1])
        Ras.append(torch.FloatTensor(Ra))
        # values could be changed
     #   values.append(torch.FloatTensor([1] * rels.shape[0]))
      #  values.append(rels[:,2])
        #### for hairs: values equals to the length of this segment
        values.append(torch.FloatTensor(vals))
        node_r_idxs.append(np.arange(n_particles))
        node_s_idxs.append(np.arange(n_particles + n_shapes))
        psteps.append(args.pstep)
        
        
    if verbose:
        print("Attr shape (after hierarchy building):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particles], axis=0))
        print("Shape attr:", np.sum(attr[n_particles:n_particles+n_shapes], axis=0))
        print("Roots attr:", np.sum(attr[n_particles+n_shapes:], axis=0))
        
        
    ### normalize data
    data = [positions, velocities]
    data = normalize(data, stat, var)
    positions, velocities = data[0], data[1]

    if verbose:
        print("Particle positions stats")
        print(positions.shape)
        print(np.min(positions[:n_particles], 0))
        print(np.max(positions[:n_particles], 0))
        print(np.mean(positions[:n_particles], 0))
        print(np.std(positions[:n_particles], 0))

        show_vel_dim = 6 if args.env == 'RiceGrip' else 3
        print("Velocities stats")
        print(velocities.shape)
        print(np.mean(velocities[:n_particles, :show_vel_dim], 0))
        print(np.std(velocities[:n_particles, :show_vel_dim], 0))
        
    state = torch.FloatTensor(np.concatenate([positions, velocities], axis=1))
    attr = torch.FloatTensor(attr)
    relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps]

    return attr, state, relations, n_particles, n_shapes#, instance_idx




class PhysicsFleXDataset(Dataset):

    def __init__(self, args, phase, phases_dict, verbose):
        self.args = args
        self.phase = phase
        self.phases_dict = phases_dict
        self.verbose = verbose
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')

        os.system('mkdir -p ' + self.data_dir)

        #    self.data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']
        self.data_names = ['positions', 'velocities','hair_idx']

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):
        return self.n_rollout * (self.args.time_step - 1)

    def load_data(self, name):
        self.stat = load_data(self.data_names[:2], self.stat_path)
        for i in range(len(self.stat)):
            self.stat[i] = self.stat[i][-self.args.position_dim:, :]
            # print(self.data_names[i], self.stat[i].shape)

    def gen_data(self, name):
        # if the data hasn't been generated, generate the data
        print("Generating data ... n_rollout=%d, time_step=%d" % (self.n_rollout, self.args.time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {
                'env': self.args.env,
                'root_num': self.phases_dict['root_num'],
                'thread_idx': i,
                'data_dir': self.data_dir,
                'data_names': self.data_names,
                'n_rollout': self.n_rollout // self.args.num_workers,
                'n_instance': self.args.n_instance,
                'time_step': self.args.time_step,
                'time_step_clip': self.args.time_step_clip,
                'dt': self.args.dt,
                'shape_state_dim': self.args.shape_state_dim}

            info['env_idx'] = 11
            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        data = pool.map(gen_PyFleX, infos)

        print("Training data generated, warpping up stats ...")

        if self.phase == 'train' and self.args.gen_stat:
            # positions [x, y, z], velocities[xdot, ydot, zdot]
            self.stat = [init_stat(3), init_stat(3)]
            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])
            store_data(self.data_names[:2], self.stat, self.stat_path)
        else:
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names[:2], self.stat_path)

    def __getitem__(self, idx):
        idx_rollout = idx // (self.args.time_step - 1)
        idx_timestep = idx % (self.args.time_step - 1)

        # ignore the first frame for env RiceGrip
        if self.args.env == 'RiceGrip' and idx_timestep == 0:
            idx_timestep = np.random.randint(1, self.args.time_step - 1)

        data_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
        data_nxt_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')

        data = load_data(self.data_names, data_path)

        '''
        vel_his = []
        for i in range(self.args.n_his):
            path = os.path.join(self.data_dir, str(idx_rollout), str(max(1, idx_timestep - i - 1)) + '.h5')
            data_his = load_data(self.data_names, path)
            vel_his.append(data_his[1])

        data[1] = np.concatenate([data[1]] + vel_his, 1)
        '''
        
        attr, state, relations, n_particles, n_shapes = \
                prepare_input(data, self.stat, self.args, self.phases_dict, self.verbose)

        ### label
        data_nxt = normalize(load_data(self.data_names, data_nxt_path), self.stat)

        label = torch.FloatTensor(data_nxt[1][:n_particles])

        return attr, state, relations, n_particles, n_shapes, label
