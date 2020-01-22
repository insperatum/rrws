import torch
import losses
import util
import itertools
print = util.print_with_time


def train_mws(generative_model, inference_network, data_loader,
              num_iterations, memory_size, callback=None):
    optimizer = torch.optim.Adam(itertools.chain(
        generative_model.parameters(), inference_network.parameters()))

    memory = {}
    data_loader_iter = iter(data_loader)

    for iteration in range(num_iterations):
        # get obs
        try:
            obs = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            obs = next(data_loader_iter)

        theta_loss = 0
        phi_loss = 0
        for single_obs in obs:
            # key to index memory
            single_obs_key = tuple(single_obs.tolist())

            # populate memory if empty
            if (
                (single_obs_key not in memory) or
                len(memory[single_obs_key]) == 0
            ):
                # batch shape [1] and event shape [num_data]
                latent_dist = inference_network.get_latent_dist(
                    single_obs.unsqueeze(0))
                # [memory_size, num_data]
                latent = inference_network.sample_from_latent_dist(
                    latent_dist, memory_size).squeeze(1)
                # list of M \in {1, ..., memory_size} elements
                # could be less than memory_size because
                # sampled elements can be duplicate
                memory[single_obs_key] = list(set(
                    [tuple(x.tolist()) for x in latent]))

            # WAKE
            # batch shape [1] and event shape [num_data]
            latent_dist = inference_network.get_latent_dist(
                single_obs.unsqueeze(0))
            # [1, 1, num_data] -> [num_data]
            latent = inference_network.sample_from_latent_dist(
                latent_dist, 1).view(-1)
            # set (of size memory_size + 1) of tuples (of length num_data)
            memoized_latent_plus_current_latent = set(
                memory.get(single_obs_key, []) +
                [tuple(latent.tolist())]
            )

            # [memory_size + 1, 1, num_data]
            memoized_latent_plus_current_latent_tensor = torch.tensor(
                list(memoized_latent_plus_current_latent),
                device=single_obs.device
            ).unsqueeze(1)
            # [memory_size + 1]
            log_p_tensor = generative_model.get_log_prob(
                memoized_latent_plus_current_latent_tensor,
                single_obs.unsqueeze(0)
            ).squeeze(-1)

            # this takes the longest
            # {int: [], ...}
            log_p = {mem_latent: lp for mem_latent, lp in zip(
                memoized_latent_plus_current_latent, log_p_tensor)}

            # update memory.
            # {float: list of ints}
            memory[single_obs_key] = sorted(
                memoized_latent_plus_current_latent,
                key=log_p.get)[-memory_size:]

            # REMEMBER
            # []
            remembered_latent_id = torch.distributions.Categorical(
                logits=torch.tensor(
                    list(map(log_p.get, memory[single_obs_key])))
            ).sample()
            remembered_latent = memory[single_obs_key][remembered_latent_id]
            remembered_latent_tensor = torch.tensor(
                remembered_latent,
                device=single_obs.device)
            # []
            theta_loss += -log_p.get(remembered_latent) / len(obs)
            # []
            phi_loss += -inference_network.get_log_prob_from_latent_dist(
                latent_dist, remembered_latent_tensor).view(()) / len(obs)

            # SLEEP
            # TODO

        optimizer.zero_grad()
        theta_loss.backward()
        phi_loss.backward()
        optimizer.step()

        print('it. {} | theta loss = {:.2f} | phi loss = {:.2f}'.format(
            iteration, theta_loss, phi_loss))

    return optimizer


def train_rws(generative_model, inference_network, data_loader,
              num_iterations, num_particles):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    optimizer_theta = torch.optim.Adam(generative_model.parameters())
    data_loader_iter = iter(data_loader)

    for iteration in range(num_iterations):
        # get obs
        try:
            obs = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            obs = next(data_loader_iter)

        log_weight, log_q = losses.get_log_weight_and_log_q(
            generative_model, inference_network, obs, num_particles)

        # wake theta
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
            log_weight)
        wake_theta_loss.backward(retain_graph=True)
        optimizer_theta.step()

        # wake phi
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
            log_weight, log_q)
        wake_phi_loss.backward()
        optimizer_phi.step()

        print('it. {} | theta loss = {:.2f} | phi loss = {:.2f}'.format(
            iteration, wake_theta_loss, wake_phi_loss))

    return optimizer_theta, optimizer_phi
