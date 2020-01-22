import torch
import losses
import util
print = util.print_with_time


def train_wake_wake(generative_model, inference_network, data_loader,
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
