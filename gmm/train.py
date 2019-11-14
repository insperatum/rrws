import torch
import losses
import util
import itertools


def train_mws(generative_model, inference_network, obss_data_loader,
              num_iterations, memory_size, callback=None):
    optimizer = torch.optim.Adam(itertools.chain(
        generative_model.parameters(), inference_network.parameters()))

    memory = {}
    # TODO: initialize memory so that for each obs, there are memory_size latents

    obss_iter = iter(obss_data_loader)

    for iteration in range(num_iterations):
        # get obss
        try:
            obss = next(obss_iter)
        except StopIteration:
            obss_iter = iter(obss_data_loader)
            obss = next(obss_iter)

        theta_loss = 0
        phi_loss = 0
        for obs in obss:
            # WAKE
            # batch shape [1] and event shape [num_mixtures]
            latent_dist = inference_network.get_latent_dist(obs.unsqueeze(0))
            # [1, 1, num_mixtures]
            latent = inference_network.sample_from_latent_dist(
                latent_dist, num_samples=1, reparam=False)
            memoized_latents_plus_current_latent = set(
                memory.get(obs.item(), []) + [latent])
            # {[1, 1, num_mixtures]: [1, 1], ...}
            log_p = {latent: generative_model.get_log_prob(latent, obs)
                     for latent in memoized_latents_plus_current_latent}

            # update memory. TODO check whether things are sorted correctly
            # {[]: list of [1, 1, num_mixtures]}
            memory[obs.item()] = sorted(memoized_latents_plus_current_latent,
                                        key=log_p.get)[-memory_size:]

            # REMEMBER
            # []
            remembered_latent_id = torch.distributions.Categorical(
                logits=torch.tensor(list(map(log_p.get, memory[obs.item()])))
            ).sample()
            # [1, 1, num_mixtures]
            remembered_latent = memory[obs.item()][remembered_latent_id]
            # []
            theta_loss += -log_p.get(remembered_latent).view(()) / len(obss)
            # []
            phi_loss += -inference_network.get_log_prob_from_latent_dist(
                latent_dist, remembered_latent).view(()) / len(obss)

            # SLEEP
            # TODO

        optimizer.zero_grad()
        theta_loss.backward()
        phi_loss.backward()
        optimizer.step()

        if callback is not None:
            callback(iteration, theta_loss.item(), phi_loss.item(),
                     generative_model, inference_network, memory, optimizer)

    return optimizer


class TrainMWSCallback():
    def __init__(self, model_folder, true_generative_model,
                 logging_interval=10, checkpoint_interval=100,
                 eval_interval=10):
        self.model_folder = model_folder
        self.true_generative_model = true_generative_model
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.test_obss = true_generative_model.sample_obs(100)

        self.theta_loss_history = []
        self.phi_loss_history = []
        self.p_error_history = []
        self.q_error_history = []

    def __call__(self, iteration, theta_loss, phi_loss,
                 generative_model, inference_network, memory, optimizer):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} losses: theta = {:.3f}, phi = {:.3f}'.format(
                    iteration, theta_loss, phi_loss))
            self.theta_loss_history.append(theta_loss)
            self.phi_loss_history.append(phi_loss)

        if iteration % self.checkpoint_interval == 0:
            stats_filename = util.get_stats_path(self.model_folder)
            util.save_object(self, stats_filename)
            util.save_models(generative_model, inference_network,
                             self.model_folder, iteration, memory)

        if iteration % self.eval_interval == 0:
            self.p_error_history.append(util.get_p_error(
                self.true_generative_model, generative_model))
            self.q_error_history.append(util.get_q_error(
                self.true_generative_model, inference_network, self.test_obss))
            util.print_with_time(
                'Iteration {} p_error = {:.3f}, q_error_to_true = '
                '{:.3f}'.format(
                    iteration, self.p_error_history[-1],
                    self.q_error_history[-1]))


def train_wake_sleep(generative_model, inference_network,
                     obss_data_loader, num_iterations, num_particles,
                     callback=None):
    num_samples = obss_data_loader.batch_size * num_particles
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    optimizer_theta = torch.optim.Adam(generative_model.parameters())
    obss_iter = iter(obss_data_loader)

    for iteration in range(num_iterations):
        # get obss
        try:
            obss = next(obss_iter)
        except StopIteration:
            obss_iter = iter(obss_data_loader)
            obss = next(obss_iter)

        # wake theta
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        wake_theta_loss, elbo = losses.get_wake_theta_loss(
            generative_model, inference_network, obss, num_particles)
        wake_theta_loss.backward()
        optimizer_theta.step()

        # sleep phi
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
        sleep_phi_loss = losses.get_sleep_loss(
            generative_model, inference_network, num_samples)
        sleep_phi_loss.backward()
        optimizer_phi.step()

        if callback is not None:
            callback(iteration, wake_theta_loss.item(), sleep_phi_loss.item(),
                     elbo.item(), generative_model, inference_network,
                     optimizer_theta, optimizer_phi)

    return optimizer_theta, optimizer_phi


class TrainWakeSleepCallback():
    def __init__(self, model_folder, true_generative_model, num_samples,
                 logging_interval=10, checkpoint_interval=100,
                 eval_interval=10):
        self.model_folder = model_folder
        self.true_generative_model = true_generative_model
        self.num_samples = num_samples
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.test_obss = true_generative_model.sample_obs(100)

        self.wake_theta_loss_history = []
        self.sleep_phi_loss_history = []
        self.elbo_history = []
        self.p_error_history = []
        self.q_error_history = []
        self.grad_std_history = []

    def __call__(self, iteration, wake_theta_loss, sleep_phi_loss, elbo,
                 generative_model, inference_network, optimizer_theta,
                 optimizer_phi):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} losses: theta = {:.3f}, phi = {:.3f}, elbo = '
                '{:.3f}'.format(iteration, wake_theta_loss, sleep_phi_loss,
                                elbo))
            self.wake_theta_loss_history.append(wake_theta_loss)
            self.sleep_phi_loss_history.append(sleep_phi_loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_filename = util.get_stats_path(self.model_folder)
            util.save_object(self, stats_filename)
            util.save_models(generative_model, inference_network,
                             self.model_folder, iteration)

        if iteration % self.eval_interval == 0:
            self.p_error_history.append(util.get_p_error(
                self.true_generative_model, generative_model))
            self.q_error_history.append(util.get_q_error(
                self.true_generative_model, inference_network, self.test_obss))
            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                sleep_phi_loss = losses.get_sleep_loss(
                    generative_model, inference_network, self.num_samples)
                sleep_phi_loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1])
            util.print_with_time(
                'Iteration {} p_error = {:.3f}, q_error_to_true = '
                '{:.3f}'.format(
                    iteration, self.p_error_history[-1],
                    self.q_error_history[-1]))


def train_wake_wake(generative_model, inference_network, obss_data_loader,
                    num_iterations, num_particles, callback=None):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    optimizer_theta = torch.optim.Adam(generative_model.parameters())
    obss_iter = iter(obss_data_loader)

    for iteration in range(num_iterations):
        # get obss
        try:
            obss = next(obss_iter)
        except StopIteration:
            obss_iter = iter(obss_data_loader)
            obss = next(obss_iter)

        log_weight, log_q = losses.get_log_weight_and_log_q(
            generative_model, inference_network, obss, num_particles)

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

        if callback is not None:
            callback(iteration, wake_theta_loss.item(), wake_phi_loss.item(),
                     elbo.item(), generative_model, inference_network,
                     optimizer_theta, optimizer_phi)

    return optimizer_theta, optimizer_phi


def train_defensive_wake_wake(delta, generative_model, inference_network,
                              obss_data_loader, num_iterations, num_particles,
                              callback=None):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    optimizer_theta = torch.optim.Adam(generative_model.parameters())
    obss_iter = iter(obss_data_loader)

    for iteration in range(num_iterations):
        # get obss
        try:
            obss = next(obss_iter)
        except StopIteration:
            obss_iter = iter(obss_data_loader)
            obss = next(obss_iter)

        log_weight, log_q = losses.get_log_weight_and_log_q(
            generative_model, inference_network, obss, num_particles)

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
        wake_phi_loss = losses.get_defensive_wake_phi_loss(
            generative_model, inference_network, obss, delta, num_particles)
        wake_phi_loss.backward()
        optimizer_phi.step()

        if callback is not None:
            callback(iteration, wake_theta_loss.item(), wake_phi_loss.item(),
                     elbo.item(), generative_model, inference_network,
                     optimizer_theta, optimizer_phi)

    return optimizer_theta, optimizer_phi


class TrainWakeWakeCallback():
    def __init__(self, model_folder, true_generative_model, num_particles,
                 logging_interval=10, checkpoint_interval=100,
                 eval_interval=10):
        self.model_folder = model_folder
        self.true_generative_model = true_generative_model
        self.num_particles = num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.test_obss = true_generative_model.sample_obs(100)

        self.wake_theta_loss_history = []
        self.wake_phi_loss_history = []
        self.elbo_history = []
        self.p_error_history = []
        self.q_error_history = []
        self.grad_std_history = []

    def __call__(self, iteration, wake_theta_loss, wake_phi_loss, elbo,
                 generative_model, inference_network, optimizer_theta,
                 optimizer_phi):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} losses: theta = {:.3f}, phi = {:.3f}, elbo = '
                '{:.3f}'.format(iteration, wake_theta_loss, wake_phi_loss,
                                elbo))
            self.wake_theta_loss_history.append(wake_theta_loss)
            self.wake_phi_loss_history.append(wake_phi_loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_filename = util.get_stats_path(self.model_folder)
            util.save_object(self, stats_filename)
            util.save_models(generative_model, inference_network,
                             self.model_folder, iteration)

        if iteration % self.eval_interval == 0:
            self.p_error_history.append(util.get_p_error(
                self.true_generative_model, generative_model))
            self.q_error_history.append(util.get_q_error(
                self.true_generative_model, inference_network, self.test_obss))
            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                wake_phi_loss = losses.get_wake_phi_loss(
                    generative_model, inference_network, self.test_obss,
                    self.num_particles)
                wake_phi_loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1])
            util.print_with_time(
                'Iteration {} p_error = {:.3f}, q_error_to_true = '
                '{:.3f}'.format(
                    iteration, self.p_error_history[-1],
                    self.q_error_history[-1]))


class TrainDefensiveWakeWakeCallback():
    def __init__(self, model_folder, true_generative_model, num_particles,
                 delta, logging_interval=10, checkpoint_interval=100,
                 eval_interval=10):
        self.model_folder = model_folder
        self.true_generative_model = true_generative_model
        self.num_particles = num_particles
        self.delta = delta
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.test_obss = true_generative_model.sample_obs(100)

        self.wake_theta_loss_history = []
        self.wake_phi_loss_history = []
        self.elbo_history = []
        self.p_error_history = []
        self.q_error_history = []
        self.grad_std_history = []

    def __call__(self, iteration, wake_theta_loss, wake_phi_loss, elbo,
                 generative_model, inference_network, optimizer_theta,
                 optimizer_phi):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} losses: theta = {:.3f}, phi = {:.3f}, elbo = '
                '{:.3f}'.format(iteration, wake_theta_loss, wake_phi_loss,
                                elbo))
            self.wake_theta_loss_history.append(wake_theta_loss)
            self.wake_phi_loss_history.append(wake_phi_loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_filename = util.get_stats_path(self.model_folder)
            util.save_object(self, stats_filename)
            util.save_models(generative_model, inference_network,
                             self.model_folder, iteration)

        if iteration % self.eval_interval == 0:
            self.p_error_history.append(util.get_p_error(
                self.true_generative_model, generative_model))
            self.q_error_history.append(util.get_q_error(
                self.true_generative_model, inference_network, self.test_obss))
            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                wake_phi_loss = losses.get_defensive_wake_phi_loss(
                    generative_model, inference_network, self.test_obss,
                    self.delta, self.num_particles)
                wake_phi_loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1])
            util.print_with_time(
                'Iteration {} p_error = {:.3f}, q_error_to_true = '
                '{:.3f}'.format(
                    iteration, self.p_error_history[-1],
                    self.q_error_history[-1]))


def train_iwae(algorithm, generative_model, inference_network,
               obss_data_loader, num_iterations, num_particles,
               callback=None):
    """Train using IWAE objective.

    Args:
        algorithm: reinforce, vimco or concrete
    """

    parameters = itertools.chain.from_iterable(
        [x.parameters() for x in [generative_model, inference_network]])
    optimizer = torch.optim.Adam(parameters)
    obss_iter = iter(obss_data_loader)

    for iteration in range(num_iterations):
        # get obss
        try:
            obss = next(obss_iter)
        except StopIteration:
            obss_iter = iter(obss_data_loader)
            obss = next(obss_iter)

        # wake theta
        optimizer.zero_grad()
        if algorithm == 'vimco':
            loss, elbo = losses.get_vimco_loss(
                generative_model, inference_network, obss, num_particles)
        elif algorithm == 'reinforce':
            loss, elbo = losses.get_reinforce_loss(
                generative_model, inference_network, obss, num_particles)
        elif algorithm == 'concrete':
            loss, elbo = losses.get_concrete_loss(
                generative_model, inference_network, obss, num_particles)
        loss.backward()
        optimizer.step()

        if callback is not None:
            callback(iteration, loss.item(), elbo.item(), generative_model,
                     inference_network, optimizer)

    return optimizer


class TrainIwaeCallback():
    def __init__(self, model_folder, true_generative_model, num_particles,
                 train_mode, logging_interval=10, checkpoint_interval=100,
                 eval_interval=10):
        self.model_folder = model_folder
        self.true_generative_model = true_generative_model
        self.num_particles = num_particles
        self.train_mode = train_mode
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.test_obss = true_generative_model.sample_obs(100)

        self.loss_history = []
        self.elbo_history = []
        self.p_error_history = []
        self.q_error_history = []
        self.grad_std_history = []

    def __call__(self, iteration, loss, elbo, generative_model,
                 inference_network, optimizer):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} loss = {:.3f}, elbo = {:.3f}'.format(
                    iteration, loss, elbo))
            self.loss_history.append(loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_filename = util.get_stats_path(self.model_folder)
            util.save_object(self, stats_filename)
            util.save_models(generative_model, inference_network,
                             self.model_folder, iteration)

        if iteration % self.eval_interval == 0:
            self.p_error_history.append(util.get_p_error(
                self.true_generative_model, generative_model))
            self.q_error_history.append(util.get_q_error(
                self.true_generative_model, inference_network, self.test_obss))
            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                if self.train_mode == 'vimco':
                    loss, elbo = losses.get_vimco_loss(
                        generative_model, inference_network, self.test_obss,
                        self.num_particles)
                elif self.train_mode == 'reinforce':
                    loss, elbo = losses.get_reinforce_loss(
                        generative_model, inference_network, self.test_obss,
                        self.num_particles)
                loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1])
            util.print_with_time(
                'Iteration {} p_error = {:.3f}, q_error_to_true = '
                '{:.3f}'.format(
                    iteration, self.p_error_history[-1],
                    self.q_error_history[-1]))


class TrainConcreteCallback():
    def __init__(self, model_folder, true_generative_model, num_particles,
                 num_iterations, logging_interval=10, checkpoint_interval=100,
                 eval_interval=10):
        self.model_folder = model_folder
        self.true_generative_model = true_generative_model
        self.num_particles = num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.test_obss = true_generative_model.sample_obs(100)

        self.loss_history = []
        self.elbo_history = []
        self.p_error_history = []
        self.q_error_history = []
        self.grad_std_history = []
        self.num_iterations = num_iterations
        self.init_temperature = 3
        self.end_temperature = 1

    def get_temperature(self, iteration):
        return self.init_temperature + iteration * \
            (self.end_temperature - self.init_temperature) / \
            self.num_iterations

    def __call__(self, iteration, loss, elbo, generative_model,
                 inference_network, optimizer):
        inference_network.set_temperature(self.get_temperature(iteration))
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} loss = {:.3f}, elbo = {:.3f}'.format(
                    iteration, loss, elbo))
            self.loss_history.append(loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_filename = util.get_stats_path(self.model_folder)
            util.save_object(self, stats_filename)
            util.save_models(generative_model, inference_network,
                             self.model_folder, iteration)

        if iteration % self.eval_interval == 0:
            self.p_error_history.append(util.get_p_error(
                self.true_generative_model, generative_model))
            self.q_error_history.append(util.get_q_error(
                self.true_generative_model, inference_network, self.test_obss))
            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                loss, _ = losses.get_concrete_loss(
                    generative_model, inference_network, self.test_obss,
                    self.num_particles)
                loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1])
            util.print_with_time(
                'Iteration {} p_error = {:.3f}, q_error_to_true = '
                '{:.3f}'.format(
                    iteration, self.p_error_history[-1],
                    self.q_error_history[-1]))


def train_relax(generative_model, inference_network, control_variate,
                obss_data_loader, num_iterations, num_particles,
                callback=None):
    """Train using RELAX."""

    num_q_params = sum([param.nelement()
                        for param in inference_network.parameters()])
    iwae_optimizer = torch.optim.Adam(itertools.chain(
        generative_model.parameters(), inference_network.parameters()))
    control_variate_optimizer = torch.optim.Adam(control_variate.parameters())
    obss_iter = iter(obss_data_loader)

    for iteration in range(num_iterations):
        # get obss
        try:
            obss = next(obss_iter)
        except StopIteration:
            obss_iter = iter(obss_data_loader)
            obss = next(obss_iter)

        # optimize theta and phi
        iwae_optimizer.zero_grad()
        control_variate_optimizer.zero_grad()
        loss, elbo = losses.get_relax_loss(
            generative_model, inference_network, control_variate, obss,
            num_particles)
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()
        loss.backward(create_graph=True)
        iwae_optimizer.step()

        # optimize rho
        control_variate_optimizer.zero_grad()
        torch.autograd.backward(
            [2 * q_param.grad / num_q_params
             for q_param in inference_network.parameters()],
            [q_param.grad.detach()
             for q_param in inference_network.parameters()]
        )
        control_variate_optimizer.step()

        if callback is not None:
            callback(iteration, loss.item(), elbo.item(), generative_model,
                     inference_network, control_variate)

    return iwae_optimizer, control_variate_optimizer


class TrainRelaxCallback():
    def __init__(self, model_folder, true_generative_model, num_particles,
                 logging_interval=10, checkpoint_interval=100,
                 eval_interval=10):
        self.model_folder = model_folder
        self.true_generative_model = true_generative_model
        self.num_particles = num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.test_obss = true_generative_model.sample_obs(100)

        self.loss_history = []
        self.elbo_history = []
        self.p_error_history = []
        self.q_error_history = []
        self.grad_std_history = []

    def __call__(self, iteration, loss, elbo, generative_model,
                 inference_network, control_variate):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} loss = {:.3f}, elbo = {:.3f}'.format(
                    iteration, loss, elbo))
            self.loss_history.append(loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_filename = util.get_stats_path(self.model_folder)
            util.save_object(self, stats_filename)
            util.save_models(generative_model, inference_network,
                             self.model_folder, iteration)
            util.save_control_variate(control_variate, self.model_folder,
                                      iteration)

        if iteration % self.eval_interval == 0:
            self.p_error_history.append(util.get_p_error(
                self.true_generative_model, generative_model))
            self.q_error_history.append(util.get_q_error(
                self.true_generative_model, inference_network, self.test_obss))
            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                loss, _ = losses.get_relax_loss(
                    generative_model, inference_network, control_variate,
                    self.test_obss, self.num_particles)
                loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1])
            util.print_with_time(
                'Iteration {} p_error = {:.3f}, q_error_to_true = '
                '{:.3f}'.format(
                    iteration, self.p_error_history[-1],
                    self.q_error_history[-1]))
