#! /usr/bin/env python3

import torch
import time
import datetime
import numpy as np
import copy
import logging
from tqdm import tqdm
import os
from pathlib import Path

from pif.influence_functions.hvp_grad import (
    grad_z,
    s_test_sample,
)
from pif.influence_functions.utils import (
    save_json,
    display_progress,
)


def calc_s_test(
        model,
        test_loader,
        train_loader,
        save=False,
        gpu=-1,
        damp=0.01,
        scale=25,
        recursion_depth=5000,
        r=1,
        start=0,
        end=0
):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    ptuples = tuple(model.named_parameters())

    s_tests = []
    for i in range(start, min(end, len(test_loader.dataset))):
        did, tid, z_test, m_test, t_test = test_loader.dataset[i]
        if save.joinpath(f"did-{int(did)}_tid-{int(tid)}_recdep{recursion_depth}_r{r}.s_test").exists():
            continue
        breakpoint()
        z_test = test_loader.collate_fn([z_test])
        m_test = test_loader.collate_fn([m_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = s_test_sample(
            model=model,
            x_test=z_test,
            m_test=m_test,
            y_test=t_test,
            train_loader=train_loader,
            gpu=gpu, damp=damp, scale=scale,
            recursion_depth=recursion_depth, r=r
        )

        if save:
            breakpoint()
            s_test_vec = [s.cpu() for s in s_test_vec]
            vecs = [vec for (n, p), vec in zip(ptuples, s_test_vec) if
                    'layer.10.' in n or 'layer.11.' in n or 'classifier.' in n or 'pooler.' in n]
            torch.save(
                vecs, save.joinpath(f"did-{int(did)}_tid-{int(tid)}_recdep{recursion_depth}_r{r}.s_test")
            )
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", i - start, end - start
        )

    return s_tests, save


def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0, end=0):
    """Calculates grad_z and can save the output to files. One grad_z should
    be computed for each training data sample.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        save_pth: Path, path where to save the grad_z files if desired.
            Omitting this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        start: int, index of the first test index to use. default is 0

    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
        save_pth: Path, path where grad_z files were saved to or
            False if they were not saved."""
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    ptuples = tuple(model.named_parameters())
    grad_zs = []
    for i in range(start, min(end, len(train_loader.dataset))):
        did, tid, z, m, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        m = train_loader.collate_fn([m])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, m, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            vecs = [vec for (n, p), vec in zip(ptuples, grad_z_vec) if
                    'layer.10.' in n or 'layer.11.' in n or 'classifier.' in n or 'pooler.' in n]
            torch.save(vecs, save_pth.joinpath(f"did-{int(did)}_tid-{int(tid)}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)
        display_progress("Calc. grad_z: ", i, len(train_loader.dataset) - i)

    return grad_zs, save_pth


def load_s_test(
        s_test_dir=Path("./s_test/"), s_test_id=0, r_sample_size=10, train_dataset_size=-1, config=None
):
    """Loads all s_test data required to calculate the influence function
    and returns a list of it.

    Arguments:
        s_test_dir: Path, folder containing files storing the s_test values
        s_test_id: int, number of the test data sample s_test was calculated
            for
        r_sample_size: int, number of s_tests precalculated
            per test dataset point
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        e_s_test: list of torch vectors, contains all e_s_tests for the whole
            dataset.
        s_test: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge."""
    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = sum(1 for _ in s_test_dir.glob("*.s_test"))
    if num_s_test_files != r_sample_size:
        logging.warning(
            "Load Influence Data: number of s_test sample files"
            " mismatches the available samples"
        )
    ########################
    # TODO: should prob. not hardcode the file name, use natsort+glob
    ########################
    #   f"{i}_recdep{recursion_depth}_r{r}.s_test"
    depth, r = config['recursion_depth'], config['r_averaging']
    for i in range(config['test_start_index'], config['test_end_index']):
        s_test.append(torch.load(str(s_test_dir) + '/' + f"{i}_recdep{depth}_r{r}.s_test"))
        display_progress("s_test files loaded: ", i, r_sample_size)

    #########################
    # TODO: figure out/change why here element 0 is chosen by default
    #########################
    e_s_test = s_test[0]
    # Calculate the sum
    for i in range(len(s_test)):
        e_s_test = [i + j for i, j in zip(e_s_test, s_test[0])]

    # Calculate the average
    #########################
    # TODO: figure out over what to calculate the average
    #       should either be r_sample_size OR e_s_test
    #########################
    e_s_test = [i / len(s_test) for i in e_s_test]

    return e_s_test, s_test


def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    """Loads all grad_z data required to calculate the influence function and
    returns it.

    Arguments:
        grad_z_dir: Path, folder containing files storing the grad_z values
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        grad_z_vecs: list of torch tensors, contains the grad_z tensors"""
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files = sum(1 for _ in grad_z_dir.glob('*.grad_z'))
    if available_grad_z_files != train_dataset_size:
        logging.warn(
            "Load Influence Data: number of grad_z files mismatches" " the dataset size"
        )
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    for i in range(train_dataset_size):
        breakpoint()
        grad_z_vecs.append(torch.load(str(grad_z_dir) + '/' + str(i) + ".grad_z"))
        display_progress("grad_z files loaded: ", i, train_dataset_size)

    return grad_z_vecs


def calc_influence_function(
        train_dataset_size,
        grad_z_vecs=None,
        e_s_test=None,
        s_test_outdir=None,
        grad_z_outdir=None,
        grad_z_test_outdir=None,
        config=None
):
    """Calculates the influence function

    Arguments:
        train_dataset_size: int, total train dataset size
        grad_z_vecs: list of torch tensor, containing the gradients
            from model parameters to loss
        e_s_test: list of torch tensor, contains s_test vectors

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness"""
    if not grad_z_vecs and not e_s_test:
        grad_z_vecs = load_grad_z(grad_z_dir=grad_z_outdir)
        grad_z_test_vecs = load_grad_z(grad_z_dir=grad_z_test_outdir)
        e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size, s_test_dir=s_test_outdir, config=config)

    if len(grad_z_vecs) != train_dataset_size:
        logging.warn(
            "Training data size and the number of grad_z files are" " inconsistent."
        )
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = (
                -sum(
                    [
                        ###################################
                        # TODO: verify if computation really needs to be done
                        # on the CPU or if GPU would work, too
                        ###################################
                        torch.sum(k * j).data.cpu().numpy()
                        for k, j in zip(grad_z_vecs[i], e_s_test)
                        ###################################
                        # Originally with [i] because each grad_z contained
                        # a list of tensors as long as e_s_test list
                        # There is one grad_z per training data sample
                        ###################################
                    ]
                )
                / train_dataset_size
        )
        influences.append(tmp_influence)
        # display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()


def calc_influence_single(
        model,
        train_loader,
        test_loader,
        test_id_num,
        gpu,
        recursion_depth,
        r,
        damp=0.01,
        scale=25,
        s_test_vec=None,
        time_logging=False,
        loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = s_test_sample(
            model,
            z_test,
            t_test,
            train_loader,
            gpu,
            recursion_depth=recursion_depth,
            r=r,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        if time_logging:
            time_a = datetime.datetime.now()

        grad_z_vec = grad_z(z, t, model, gpu=gpu)

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )
        with torch.no_grad():
            tmp_influence = (
                    -sum(
                        [
                            ####################
                            # TODO: potential bottle neck, takes 17% execution time
                            # torch.sum(k * j).data.cpu().numpy()
                            ####################
                            torch.sum(k * j).data
                            for k, j in zip(grad_z_vec, s_test_vec)
                        ]
                    )
                    / train_dataset_size
            )

        influences.append(tmp_influence)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num


def get_dataset_sample_ids_per_class(class_id, num_samples, test_loader, start_index=0):
    """Gets the first num_samples from class class_id starting from
    start_index. Returns a list with the indicies which can be passed to
    test_loader.dataset[X] to retreive the actual data.

    Arguments:
        class_id: int, name or id of the class label
        num_samples: int, number of samples per class to process
        test_loader: DataLoader, can load the test dataset.
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_list: list of int, contains indicies of the relevant samples"""
    sample_list = []
    img_count = 0
    for i in range(len(test_loader.dataset)):
        _, t = test_loader.dataset[i]
        if class_id == t:
            img_count += 1
            if (img_count > start_index) and (img_count <= start_index + num_samples):
                sample_list.append(i)
            elif img_count > start_index + num_samples:
                break

    return sample_list


def get_dataset_sample_ids(num_samples, test_loader, num_classes=None, start_index=0):
    """Gets the first num_sample indices of all classes starting from
    start_index per class. Returns a list and a dict containing the indicies.

    Arguments:
        num_samples: int, number of samples of each class to return
        test_loader: DataLoader, can load the test dataset
        num_classes: int, number of classes contained in the dataset
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_dict: dict, containing dict[class] = list_of_indices
        sample_list: list, containing a continious list of indices"""
    sample_dict = {}
    sample_list = []
    if not num_classes:
        num_classes = len(np.unique(test_loader.dataset.targets))
    for i in range(num_classes):
        sample_dict[str(i)] = get_dataset_sample_ids_per_class(
            i, num_samples, test_loader, start_index
        )
        # Append the new list on the same level as the old list
        # Avoids having a list of lists
        sample_list[len(sample_list): len(sample_list)] = sample_dict[str(i)]
    return sample_dict, sample_list


def calc_img_wise(config, model, train_loader, test_loader, loss_func="cross_entropy"):
    """Calculates the influence function one test point at a time. Calcualtes
    the `s_test` and `grad_z` values on the fly and discards them afterwards.

    Arguments:
        config: dict, contains the configuration from cli params"""
    influences_meta = copy.deepcopy(config)
    test_sample_num = config["test_sample_num"]
    test_start_index = config["test_start_index"]
    outdir = Path(config["outdir"])

    # If calculating the influence for a subset of the whole dataset,
    # calculate it evenly for the same number of samples from all classes.
    # `test_start_index` is `False` when it hasn't been set by the user. It can
    # also be set to `0`.
    if test_sample_num and test_start_index is not False:
        test_dataset_iter_len = test_sample_num * config["num_classes"]
        _, sample_list = get_dataset_sample_ids(
            test_sample_num, test_loader, config["num_classes"], test_start_index
        )
    else:
        test_dataset_iter_len = len(test_loader.dataset)

    # Set up logging and save the metadata conf file
    logging.info(f"Running on: {test_sample_num} images per class.")
    logging.info(f"Starting at img number: {test_start_index} per class.")
    influences_meta["test_sample_index_list"] = sample_list
    influences_meta_fn = (
        f"influences_results_meta_{test_start_index}-" f"{test_sample_num}.json"
    )
    influences_meta_path = outdir.joinpath(influences_meta_fn)
    save_json(influences_meta, influences_meta_path)

    influences = {}
    # Main loop for calculating the influence function one test sample per
    # iteration.
    for j in range(test_dataset_iter_len):
        # If we calculate evenly per class, choose the test img indicies
        # from the sample_list instead
        if test_sample_num and test_start_index:
            if j >= len(sample_list):
                logging.warning(
                    "ERROR: the test sample id is out of index of the"
                    " defined test set. Jumping to next test sample."
                )
            i = sample_list[j]
        else:
            i = j

        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(
            model,
            train_loader,
            test_loader,
            test_id_num=i,
            gpu=config["gpu"],
            recursion_depth=config["recursion_depth"],
            r=config["r_averaging"],
            loss_func=loss_func,
        )
        end_time = time.time()

        ###########
        # Different from `influence` above
        ###########
        influences[str(i)] = {}
        _, label = test_loader.dataset[i]
        influences[str(i)]["label"] = label
        influences[str(i)]["num_in_dataset"] = j
        influences[str(i)]["time_calc_influence_s"] = end_time - start_time
        infl = [x.cpu().numpy().tolist() for x in influence]
        influences[str(i)]["influence"] = infl
        influences[str(i)]["harmful"] = harmful[:500]
        influences[str(i)]["helpful"] = helpful[:500]

        tmp_influences_path = outdir.joinpath(
            f"influence_results_tmp_"
            f"{test_start_index}_"
            f"{test_sample_num}"
            f"_last-i_{i}.json"
        )
        save_json(influences, tmp_influences_path)
        display_progress("Test samples processed: ", j, test_dataset_iter_len)

    logging.info(f"The results for this run are:")
    logging.info("Influences: ")
    logging.info(influence[:3])
    logging.info("Most harmful img IDs: ")
    logging.info(harmful[:3])
    logging.info("Most helpful img IDs: ")
    logging.info(helpful[:3])

    influences_path = outdir.joinpath(
        f"influence_results_{test_start_index}_" f"{test_sample_num}.json"
    )
    save_json(influences, influences_path)


def calc_all_grad_then_test(config, model, train_loader, test_loader, calculate_if=False):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    '''
    config['outdir'] specifies the model name, i.e. different model saves at different out dir with unique name/id/
    
    '''
    outdir = Path(config["outdir"])
    if not outdir.exists():
        outdir.mkdir()
    s_test_outdir = outdir.joinpath("s_test/")
    if not s_test_outdir.exists():
        s_test_outdir.mkdir()
    grad_z_outdir = outdir.joinpath("grad_z/")
    if not grad_z_outdir.exists():
        grad_z_outdir.mkdir()

    grad_z_test_outdir = outdir.joinpath("grad_z_test/")
    if not grad_z_test_outdir.exists():
        grad_z_test_outdir.mkdir()

    influence_results = {}
    # import pdb; pdb.set_trace()
    calc_s_test(
        model,
        test_loader,
        train_loader,
        s_test_outdir,
        gpu=config["gpu"],
        damp=config["damp"],
        scale=config["scale"],
        recursion_depth=config["recursion_depth"],
        r=config["r_averaging"],
        start=config["test_hessian_start_index"],
        end=config['test_hessian_end_index']
    )
    breakpoint()
    calc_grad_z(
        model, train_loader, grad_z_outdir, config["gpu"],
        start=config["train_start_index"],
        end=config['train_end_index']
    )
    breakpoint()
    calc_grad_z(
        model, test_loader, grad_z_test_outdir, config["gpu"],
        start=config["test_start_index"],
        end=config['test_end_index']
    )
    breakpoint()


def calc_all_grad_then_test_mask(config, model, Utrain_loader, train_loader, test_loader, calculate_if=False):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    '''
    config['outdir'] specifies the model name, i.e. different model saves at different out dir with unique name/id/
    
    '''
    outdir = Path(config["outdir"])
    if not outdir.exists():
        outdir.mkdir()
    s_test_outdir = outdir.joinpath("s_test/")
    if not s_test_outdir.exists():
        s_test_outdir.mkdir()
    grad_z_outdir = outdir.joinpath("grad_z/")
    if not grad_z_outdir.exists():
        grad_z_outdir.mkdir()

    grad_z_test_outdir = outdir.joinpath("grad_z_test/")
    if not grad_z_test_outdir.exists():
        grad_z_test_outdir.mkdir()

    influence_results = {}
    # import pdb; pdb.set_trace()
    calc_s_test(
        model,
        test_loader,
        Utrain_loader,
        s_test_outdir,
        gpu=config["gpu"],
        damp=config["damp"],
        scale=config["scale"],
        recursion_depth=config["recursion_depth"],
        r=config["r_averaging"],
        start=config["test_hessian_start_index"],
        end=config['test_hessian_end_index']
    )
    breakpoint()
    calc_grad_z(
        model, train_loader, grad_z_outdir, config["gpu"],
        start=config["train_start_index"],
        end=config['train_end_index']
    )
    breakpoint()
    calc_grad_z(
        model, test_loader, grad_z_test_outdir, config["gpu"],
        start=config["test_start_index"],
        end=config['test_end_index']
    )
    breakpoint()


def create_map_from_dataloader(dataloader):
    id_data_map = {}
    for i in range(len(dataloader.dataset)):
        did, tid, z, m, t = dataloader.dataset[i]
        did, tid = int(did), int(tid)
        if did not in id_data_map: id_data_map[did] = {}
        id_data_map[did][tid] = (z, m, t)
    return id_data_map


def calc_all_grad_mask(config, model,
                       train_loader, test_loader,
                       Mtrain_loader, Mtest_loader,
                       mode,
                       traindata, testdata,
                       Mtraindata, Mtestdata,
                       ntest_start, ntest_end,
                       test_delta_flag=True
                       ):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    '''
    config['outdir'] specifies the model name, i.e. different model saves at different out dir with unique name/id/
    
    '''
    depth, r = config['recursion_depth'], config['r_averaging']

    outdir = Path(config["outdir"])
    breakpoint()
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    mask_train_id_data_map = create_map_from_dataloader(Mtrain_loader)
    mask_test_id_data_map = create_map_from_dataloader(Mtest_loader)

    influence_results = {}

    ntrainiter = len(train_loader.dataset)

    for i in tqdm(range(ntest_start, ntest_end)):
        did, tid, z, m, t = test_loader.dataset[i]
        did = int(did)
        tid = int(tid)
        if test_delta_flag:
            if outdir.joinpath(f'did-{did}_tid-{tid}.{mode}.json').exists(): continue
        else:
            if outdir.joinpath(f'did-{did}_tid-{tid}.{mode}_test_only.json').exists(): continue

        if mode == 'TC-MASK':
            z = test_loader.collate_fn([z])
            m = test_loader.collate_fn([m])
            t = test_loader.collate_fn([t])
            grad_z_test = grad_z(z, m, t, model, gpu=config['gpu'])
            grad_z_test = pick_gradient(grad_z_test, model)  # pick effective ones

            (zmasked, mmasked, tmasked) = mask_test_id_data_map[did][tid]

            zmasked = Mtest_loader.collate_fn([zmasked])
            mmasked = Mtest_loader.collate_fn([mmasked])
            tmasked = Mtest_loader.collate_fn([tmasked])
            grad_z_test_masked = grad_z(zmasked, mmasked, tmasked, model, gpu=config['gpu'])
            grad_z_test_masked = pick_gradient(grad_z_test_masked, model)

        if mode == 'IF-MASK':
            if not os.path.exists(
                config['stest_path'] + f"/did-{int(did)}_tid-{int(tid)}_recdep{depth}_r{r}.s_test"): continue
            s_test = torch.load(config['stest_path'] + f"/did-{int(did)}_tid-{int(tid)}_recdep{depth}_r{r}.s_test")
            if not os.path.exists(
                config['stest_mask_path'] + f"/did-{int(did)}_tid-{int(tid)}_recdep{depth}_r{r}.s_test"): continue
            s_test_masked = torch.load(
                config['stest_mask_path'] + f"/did-{int(did)}_tid-{int(tid)}_recdep{depth}_r{r}.s_test")
            s_test = [st.cuda() for st in s_test]
            s_test_masked = [st.cuda() for st in s_test_masked]

        train_influences = {}

        for j in tqdm(range(ntrainiter)):
            tdid, ttid, tz, tm, tt = train_loader.dataset[j]
            tdid = int(tdid)
            ttid = int(ttid)
            tz = train_loader.collate_fn([tz])
            tm = train_loader.collate_fn([tm])
            tt = train_loader.collate_fn([tt])
            grad_z_train = grad_z(tz, tm, tt, model, gpu=config['gpu'])
            grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones

            (ztrainmasked, mtrainmasked, ttrainmasked) = mask_train_id_data_map[tdid][ttid]
            ztrainmasked = Mtrain_loader.collate_fn([ztrainmasked])
            mtrainmasked = Mtrain_loader.collate_fn([mtrainmasked])
            ttrainmasked = Mtrain_loader.collate_fn([ttrainmasked])
            grad_z_train_masked = grad_z(ztrainmasked, mtrainmasked, ttrainmasked, model, gpu=config['gpu'])
            grad_z_train_masked = pick_gradient(grad_z_train_masked, model)

            grad_z_train_delta = [gzt - gztm for gzt, gztm in zip(grad_z_train, grad_z_train_masked)]
            if mode == 'IF-MASK':
                if test_delta_flag:
                    s_test_delta = [s_t - s_tm for s_t, s_tm in zip(s_test, s_test_masked)]
                else:
                    s_test_delta = s_test
                score = param_vec_dot_product(s_test_delta, grad_z_train_delta)

            elif mode == 'TC-MASK':
                if test_delta_flag:
                    grad_z_test_delta = [gzt - gztm for gzt, gztm in zip(grad_z_test, grad_z_test_masked)]
                else:
                    grad_z_test_delta = grad_z_test

                score = param_vec_dot_product(grad_z_test_delta, grad_z_train_delta)

            breakpoint()

            if tdid not in train_influences:
                train_influences[tdid] = {}
            if ttid not in train_influences[tdid]:
                train_influences[tdid][ttid] = {'train_dat': traindata[tdid][ttid], 'if': float(score)}

        if did not in influence_results: influence_results[did] = {}
        if tid not in influence_results[did]: influence_results[did][tid] = None
        influence_results[did][tid] = {'test_dat': testdata[did][tid], 'ifs': train_influences}
        if test_delta_flag:
            save_json(influence_results, outdir.joinpath(f'did-{did}_tid-{tid}.{mode}.json'))
        else:
            save_json(influence_results, outdir.joinpath(f'did-{did}_tid-{tid}.{mode}_test_only.json'))


def calc_all_grad(config, model, train_loader, test_loader,
                  ntest_start, ntest_end, mode='TC'):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    '''
    config['outdir'] specifies the model name, i.e. different model saves at different out dir with unique name/id/
    
    '''
    depth, r = config['recursion_depth'], config['r_averaging']

    outdir = Path(config["outdir"])
    breakpoint()
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    influence_results = {}

    ntrainiter = len(train_loader.dataset)

    for i in tqdm(range(ntest_start, ntest_end)):
        idx, input_ids, token_type_ids, label = test_loader.dataset[i]

        idx = int(idx)

        if outdir.joinpath(f'did-{idx}.{mode}.json').exists():
            continue

        if mode == 'TC':
            z = test_loader.collate_fn([input_ids])
            m = test_loader.collate_fn([token_type_ids])
            t = test_loader.collate_fn([label])
            grad_z_test = grad_z(z, m, t, model, gpu=config['gpu'])
            grad_z_test = pick_gradient(grad_z_test, model)  # pick effective ones

        if mode == 'IF':
            s_test = torch.load(config['stest_path'] + f"/did-{int(idx)}_recdep{depth}_r{r}.s_test")
            s_test = [s_t.cuda() for s_t in s_test]

        train_influences = {}

        for j in tqdm(range(ntrainiter)):
            tidx, tz, tm, tt = train_loader.dataset[j]
            tidx = int(tidx)
            tz = train_loader.collate_fn([tz])
            tm = train_loader.collate_fn([tm])
            tt = train_loader.collate_fn([tt])
            grad_z_train = grad_z(tz, tm, tt, model, gpu=config['gpu'])
            grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones
            if mode == 'IF':
                score = param_vec_dot_product(s_test, grad_z_train)
            elif mode == 'TC':
                score = param_vec_dot_product(grad_z_test, grad_z_train)

            breakpoint()

            if tidx not in train_influences:
                train_influences[tidx] = {'train_dat': (tz, tm, tt),
                                          'if': float(score)}

        if idx not in influence_results:
            influence_results[idx] = {'test_dat': (input_ids, token_type_ids, label),
                                      'ifs': train_influences}

        save_json(influence_results, outdir.joinpath(f'did-{idx}.{mode}.json'))


def param_vec_dot_product(a, b):
    ''' dot product between two lists'''
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])

    breakpoint()


def pick_gradient(grads, model):
    """
    pick the gradients by name.
    Specifically for BERTs, it extracts 10, 11 layer, pooler and classification layers params.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())
            if 'layer.10.' in n or 'layer.11.' in n
            or 'classifier.' in n or 'pooler.' in n]
