import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import corner
import graphviz
import matplotlib.pyplot as plt

def plot_df(df):
    plt.clf()
    plt.scatter(df[df['planet_code']==0]['theta'].values, df[df['planet_code']==0]['pos'].values,
                color = '#FF220C', label = 'planet 0')
    plt.errorbar(df[df['planet_code']==0]['theta'].values, df[df['planet_code']==0]['pos'].values,
                yerr = df[df['planet_code']==0]['pos_err'].values, linestyle = 'None',
                color = '#FF220C')
    plt.scatter(df[df['planet_code']==1]['theta'].values, df[df['planet_code']==1]['pos'].values,
                color = '#9B7874', label = 'planet 1')
    plt.errorbar(df[df['planet_code']==1]['theta'].values, df[df['planet_code']==1]['pos'].values,
                yerr = df[df['planet_code']==1]['pos_err'].values, linestyle = 'None',
                color = '#9B7874')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel('x pos')
    plt.legend()
    plt.show()

    plt.clf()
    plt.scatter(df[df['planet_code']==0]['length'].values, df[df['planet_code']==0]['pos'].values,
                color = '#FF220C', label = 'planet 0')
    plt.errorbar(df[df['planet_code']==0]['length'].values, df[df['planet_code']==0]['pos'].values,
                yerr = df[df['planet_code']==0]['pos_err'].values, linestyle = 'None',
                color = '#FF220C')
    plt.scatter(df[df['planet_code']==1]['length'].values, df[df['planet_code']==1]['pos'].values,
                color = '#9B7874', label = 'planet 1')
    plt.errorbar(df[df['planet_code']==1]['length'].values, df[df['planet_code']==1]['pos'].values,
                yerr = df[df['planet_code']==1]['pos_err'].values, linestyle = 'None',
                color = '#9B7874')
    plt.xlabel(r'length')
    plt.ylabel('x pos')
    plt.legend()
    plt.show()

    plt.clf()
    plt.scatter(df[df['planet_code']==0]['a_g'].values, df[df['planet_code']==0]['pos'].values,
                color = '#FF220C', label = 'planet 0')
    plt.errorbar(df[df['planet_code']==0]['a_g'].values, df[df['planet_code']==0]['pos'].values,
                yerr = df[df['planet_code']==0]['pos_err'].values, linestyle = 'None',
                color = '#FF220C')
    plt.scatter(df[df['planet_code']==1]['a_g'].values, df[df['planet_code']==1]['pos'].values,
                color = '#9B7874', label = 'planet 1')
    plt.errorbar(df[df['planet_code']==1]['a_g'].values, df[df['planet_code']==1]['pos'].values,
                yerr = df[df['planet_code']==1]['pos_err'].values, linestyle = 'None',
                color = '#9B7874')
    plt.xlabel(r'$a_g$')
    plt.ylabel('x pos')
    plt.legend()
    plt.show()

def compare_corner(posterior_samples_hierarchical,
                   posterior_samples_unpooled):
    data = az.from_dict(
            posterior={"ag0": posterior_samples_hierarchical["a_g"][:,0], "ag1": posterior_samples_hierarchical["a_g"][:,1],
                       #"L2": posterior["L"][:,2], "L3": posterior["L"][:,3],
                       #"L4": posterior["L"][:,4], "L5": posterior["L"][:,5],
                       #"L6": posterior["L"][:,6], "L7": posterior["L"][:,7],
                       },
            #sample_stats={"diverging": posterior["L"][:,0] < 9.0},
    )
    plt.clf()
    figure = corner.corner(data, truths = [data_params['a_g'][0], data_params['a_g'][-1]], truth_color = '#D84797')
    plt.show()

    data = az.from_dict(
                posterior={"ag0": posterior_samples_unpooled["a_g"][:,0], "ag1": posterior_samples_unpooled["a_g"][:,1],
                        #"L2": posterior["L"][:,2], "L3": posterior["L"][:,3],
                        #"L4": posterior["L"][:,4], "L5": posterior["L"][:,5],
                        #"L6": posterior["L"][:,6], "L7": posterior["L"][:,7],
                        },
                #sample_stats={"diverging": posterior["L"][:,0] < 9.0},
    )
    plt.clf()
    figure = corner.corner(data, truths = [data_params['a_g'][0], data_params['a_g'][-1]], truth_color = '#D84797')
    plt.show()

def az_trace(data):
    plt.clf()
    az.plot_trace(data, compact=True, figsize=(15, 25), legend=True)
    plt.show()

def sampling_summary_table(mcmc):
    ## the arviz tools allows us to investigate the chain performance
    inf_data = az.from_numpyro(mcmc)
    ## zero divergence means energy is conserved
    print(f'divergences: {inf_data.sample_stats.diverging.values.sum()}')
    return az.summary(inf_data), inf_data

## plotting and analysis utilities

def display_pendulum_data(df):
    ## plot all pendulums and pendulums color-coded by planet
    color_list = ['#BCF4F5', '#B4EBCA', '#D9F2B4', '#D3FAC7',
                  '#FFB7C3', '#2F2F2F', '#4A4063', '#FE5E41']
    index = 0
    for pend in np.unique(df['pendulum_id']):
        subset = df[df['pendulum_id']==pend]
        plt.plot(subset['time'], subset['pos'],
                    color = color_list[index],
                    label = pend)
        plt.scatter(subset['time'], subset['pos'],
                    color = color_list[index])
        index+=1
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('x position')
    plt.show()

    index = 0
    for planet in np.unique(df['planet_id']):
        subset = df[df['planet_id']==planet]
        plt.plot(subset['time'], subset['pos'],
                    color = color_list[index],
                    label = planet)
        plt.scatter(subset['time'], subset['pos'],
                    color = color_list[index])
        if planet > 3:
            break
        index+=1
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('x position')
    plt.title('pendulums color-coded by planet')
    plt.show()
    
def plot_prior_predictive(prior_pred,
                          df,
                          variable_model,
                          variable_df,
                          n_steps=10,
                          title = None):
    ## plot the prior predictive histograms for parameters in the model
    label = "prior samples"
    plt.hist(
        prior_pred[variable_model].flatten(),
        n_steps,
        #range=(0.2, 3.0),
        #histtype="step",
        color="k",
        lw=0.5,
        alpha=0.5,
        label=label,
        density=True
    )
    plt.hist(df[variable_df], n_steps, histtype="step", color="black", label="data", density=True)
    plt.legend()
    plt.xlabel(variable_model)
    plt.ylabel("density")
    plt.title(title)
    plt.show()

def pos_calculator(L, theta, a_g, sigma, time):
    ## because the sampler chain doesn't save the position at each point in time,
    ## i have created a thing that samples position for us
    pos = L * np.sin(theta * np.cos(np.sqrt(a_g / L) * time)) + sigma * np.random.randn(len(time))
    return pos

def how_did_we_do_on_individual_pendulums(df, posterior, n_pendulums, chain_length, pooled = False):
    ## print out the mean and standard deviation on the posterior parameters for
    ## each individual pendulum and make multiple draws from the posterior distribution
    ## to plot "theoretical" pendulum trajectories
    for number in range(n_pendulums):
        sub_df = df[df['pend_code'] == number]
        print(f"Pendulum number {number}")
        print(f"true L = {np.mean(sub_df['length'].values)}")
        print(f"posterior L = {round(np.median(posterior['L'][:,number]),2)} +/- {round(np.std(posterior['L'][:,number]),2)}")
        print(f"true theta = {np.mean(sub_df['theta'].values)}")
        print(f"posterior theta = {round(np.median(posterior['theta'][:,number]),2)} +/- {round(np.std(posterior['theta'][:,number]),2)}")
        print(f"true a_g = {np.mean(sub_df['a_g'].values)}")
        if pooled:
            print(f"posterior a_g = {round(np.median(posterior['a_g'][:]),2)} +/- {round(np.std(posterior['a_g'][:]),2)}")
        else:
            print(f"posterior a_g = {round(np.median(posterior['a_g'][:,number]),2)} +/- {round(np.std(posterior['a_g'][:,number]),2)}")
        # let's actually draw from this posterior:
        plt.clf()

        
        for j in range(chain_length):
            L = posterior['L'][j,number]
            theta = posterior['theta'][j,number]
            if pooled:
                a_g = posterior['a_g'][j]
            else:
                a_g = posterior['a_g'][j,number]#FIXXXXX
            sigma = posterior['σ'][j]
            plt.plot(times, pos_calculator(L,theta,a_g,sigma,times), color = 'grey')
        plt.scatter(sub_df['time'].values, sub_df['pos'].values, zorder=100, color = 'orange')
        plt.show()

def how_did_we_do_on_all_pendulums(df, posterior, n_pendulums, chain_length):
    ## plot the 3 sigma position intervals from these posteriors for all pendulums at once
    ## (should probably redo this to more cleanly see each pendulum individually)
    n_pendulums = 8
    chain_length = 2000
    # first do the first four pendulums
    # hot colors
    colors = ['#F26419', '#F6AE2D', '#820263', '#D90368']
    plt.clf()

    offset = 0
    counter_neg = 0
    for number in range(4):
        sub_df = df[df['pend_code'] == number]
        # let's actually draw from this posterior:


        y_model = np.zeros((chain_length, len(times)))

        # I need to grab this from the end of the chain (most accurate)
        for j in range(chain_length):
            L = posterior['L'][j,number]
            theta = posterior['theta'][j,number]
            try:
                a_g = posterior['a_g'][j,number]#FIXXXXX
            except IndexError: # for the unpooled case
                a_g = posterior['a_g'][j]
            sigma = posterior['σ'][j]
            # Instead of plotting can I combine the lines?
            if L < 0 or a_g < 0:
                counter_neg += 1
            y_model[j,:] = pos_calculator(abs(L),theta,abs(a_g),sigma,times)
            #if j > 100:
            #    break
        y_average = np.mean(y_model, axis=0)
        y_std = np.std(y_model, axis=0)
        plt.fill_between(times, y_average + offset - 3 * y_std, y_average + offset + 3 * y_std, color=colors[number],
                         alpha=0.5, label='Standard Deviation')
        #plt.plot(times, y_average + offset, color = colors[number])
        #plt.scatter(sub_df['time'].values, sub_df['pos'].values + offset, zorder=100, color = colors[number])
        plt.axhline(y = offset, color = 'black')
        offset += 1 #* (number + 1)
        
    print('NUMER OF NEGATIVE PARAMS', counter_neg)
    plt.title('Pendulums on planet 1')
    plt.ylabel('each subsequent pendulum is offset by +15')
    plt.show()
    

    # second four pendulums (from planet 2) are cool colors
    colors = ['#33658A', '#86BBD8', '#2F4858', '#6CC551']
    plt.clf()
    offset = 0
    for number in range(4):
        number_2 = number + 3
        sub_df = df[df['pend_code'] == number_2]
        # let's actually draw from this posterior:


        y_model = np.zeros((chain_length, len(times)))

        # I need to grab this from the end of the chain (most accurate)
        for j in range(chain_length):
            L = posterior['L'][j,number_2]
            theta = posterior['theta'][j,number_2]
            try:
                a_g = posterior['a_g'][j,number_2]#
            except IndexError:
                a_g = posterior['a_g'][j]
            sigma = posterior['σ'][j]
            # Instead of plotting can I combine the lines?
            y_model[j,:] = pos_calculator(abs(L),theta,abs(a_g),sigma,times)
            #if j > 100:
            #    break
        y_average = np.mean(y_model, axis=0)
        y_std = np.std(y_model, axis=0)
        plt.fill_between(times, y_average + offset - 3 * y_std, y_average + offset + 3 * y_std, color=colors[number],
                         alpha=0.5, label='Standard Deviation')
        plt.plot(times, y_average + offset, color = colors[number])
        plt.scatter(sub_df['time'].values, sub_df['pos'].values + offset, zorder=100, color = colors[number])
        plt.axhline(y = offset, color = 'black')
        offset += 1
    plt.title('Pendulums on planet 2')
    plt.ylabel('each subsequent pendulum is offset by +15')
    plt.show()

def make_corner_plots(posterior, data_params, pooled = False):
    data = az.from_dict(
        posterior={"L0": posterior["L"][:,0], "L1": posterior["L"][:,1],
                   "L2": posterior["L"][:,2], "L3": posterior["L"][:,3],
                   "L4": posterior["L"][:,4], "L5": posterior["L"][:,5],
                   "L6": posterior["L"][:,6], "L7": posterior["L"][:,7],
                   },
   #     sample_stats={"diverging": posterior["L"][:,0] < 9.0},
    )
    plt.clf()
    figure = corner.corner(data, divergences=True, truths = data_params['length'], truth_color = '#D84797')#,
                           #range = [(5,20),(5,20),(5,20),(5,20),(5,20),(5,20),(5,20),(5,20)])
    plt.show()

    data = az.from_dict(
        posterior={"theta0": posterior["theta"][:,0], "theta1": posterior["theta"][:,1],
                   "theta2": posterior["theta"][:,2], "theta3": posterior["theta"][:,3],
                   "theta4": posterior["theta"][:,4], "theta5": posterior["theta"][:,5],
                   "theta6": posterior["theta"][:,6], "theta7": posterior["theta"][:,7],
                   },
        #sample_stats={"diverging": posterior["L"][:,0] < 9.0},
    )
    plt.clf()
    figure = corner.corner(data, truths = data_params['theta'], truth_color = '#D84797')
    plt.show()

    if pooled:
        plt.clf()
        plt.hist(posterior["a_g"], bins = 100)
        plt.axvline(x = data_params['a_g'][0])
        plt.axvline(x = data_params['a_g'][4])
        plt.show()
    else:
        data = az.from_dict(
            posterior={"ag0": posterior["a_g"][:,0], "ag1": posterior["a_g"][:,1],
                       #"L2": posterior["L"][:,2], "L3": posterior["L"][:,3],
                       #"L4": posterior["L"][:,4], "L5": posterior["L"][:,5],
                       #"L6": posterior["L"][:,6], "L7": posterior["L"][:,7],
                       },
            #sample_stats={"diverging": posterior["L"][:,0] < 9.0},
        )
        print(data_params['a_g'])
        plt.clf()
        figure = corner.corner(data, truths = [data_params['a_g'][0], data_params['a_g'][4]], truth_color = '#D84797')
        plt.show()
    data = az.from_dict(
        posterior={"ag0": posterior["a_g"][:,0], "ag1": posterior["a_g"][:,1],
                   "L0": posterior["L"][:,0], "L1": posterior["L"][:,1],
                   "L4": posterior["L"][:,4], "L5": posterior["L"][:,5],
                   #"L6": posterior["L"][:,6], "L7": posterior["L"][:,7],
                   },
        #sample_stats={"diverging": posterior["L"][:,0] < 9.0},
    )
    plt.clf()
    figure = corner.corner(data, truths = [data_params['a_g'][0], data_params['a_g'][4],
                                           data_params['length'][0], data_params['length'][1],
                                           data_params['length'][4], data_params['length'][5]], truth_color = '#D84797')
    plt.show()
    
def plot_posterior_predictive_samples(df, model, posterior, rng_key, time_values, mcmc_run,
                                      pooled,
                                      title,
                                      kind = 'kde'):
    
    planet_encoder = LabelEncoder()

    planets = planet_encoder.fit_transform(df["planet_code"])
    planets = jnp.array(planets)

    pendulum_encoder = LabelEncoder()

    pendulums = pendulum_encoder.fit_transform(df["pend_code"])
    pendulums = jnp.array(pendulums)

    pooled_posterior_predictive = numpyro.infer.Predictive(
        model=model, posterior_samples=posterior
    )
    '''
    prior_pred = numpyro.infer.Predictive(hierarchical_model, num_samples=50)(
        random.PRNGKey(11), planet_code, pend_code, time_values
    )
    '''
    rng_key, rng_subkey = random.split(rng_key)
    if pooled:
        print('pooled')
        pooled_posterior_predictive_samples = pooled_posterior_predictive(rng_subkey, pendulums, time_values)
    else:
        pooled_posterior_predictive_samples = pooled_posterior_predictive(rng_subkey, planets, pendulums, time_values)
    # convert to arviz inference data object
    pooled_idata = az.from_numpyro(
        posterior=mcmc_run, posterior_predictive=pooled_posterior_predictive_samples
    )
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_ppc(
        data=pooled_idata,
        observed_rug=True,
        ax=ax,
        kind=kind,
    )
    ax.set(
        title=title,
        xlabel="observed position",
        ylabel="count",
    )
    plt.show()
    
    
    return pooled_posterior_predictive, pooled_posterior_predictive_samples, pooled_idata
    
def compare_models(pooled_idata, unpooled_idata, hierarchical_idata, kind = 'kde'):
    plt.clf()
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 5), sharex=True, sharey=True#, layout="constrained"
    )

    az.plot_ppc(
        data=pooled_idata,
        observed_rug=True,
        ax=ax[0],
        kind=kind,
    )
    ax[0].set(
        title="Pooled Model",
        xlabel="x pos",
        ylabel="count",
    )
    az.plot_ppc(
        data=unpooled_idata,
        observed_rug=True,
        ax=ax[1],
        kind=kind,
    )
    ax[1].set(
        title="Unpooled",
        xlabel="x pos",
        ylabel="count",
    )
    az.plot_ppc(
        data=hierarchical_idata,
        observed_rug=True,
        ax=ax[2],
        kind=kind,
    )
    ax[2].set(
        title="Hierarchical",
        xlabel="x pos",
        ylabel="count",
    )

    fig.suptitle("Posterior Predictive Checks", y=1.06, fontsize=16)
    plt.show()
    
def investigate_distribution_posteriors(posterior_samples, inf_data):
    ## investigate the distribution
    for num in range(8):
        heights, bins = np.histogram(posterior_samples['L'][:,num], bins = 100)
        height = np.max(heights)
        p3 = np.percentile(posterior_samples['L'][:,num], 3)
        p97 = np.percentile(posterior_samples['L'][:,num], 97)
        plt.hist(posterior_samples['L'][:,num], color = '#D84797', bins = 100)
        plt.axvline(x = az.summary(inf_data)['mean'][num], color = 'black')
        plt.text(az.summary(inf_data)['mean'][num], height, 'mean', color='black', ha='right', va='top', rotation=90)
        plt.axvline(x = az.summary(inf_data)['hdi_3%'][num], color = 'black')
        plt.text(az.summary(inf_data)['hdi_3%'][num], height, 'hdi_3%', color='black', ha='right', va='top', rotation=90)
        plt.axvline(x = az.summary(inf_data)['hdi_97%'][num], color = 'black')
        plt.text(az.summary(inf_data)['hdi_97%'][num], height, 'hdi_97%', color='black', ha='right', va='top', rotation=90)
        plt.axvline(x = az.summary(inf_data)['mean'][num] - az.summary(inf_data)['sd'][num], color = 'black')
        plt.text(az.summary(inf_data)['mean'][num] - az.summary(inf_data)['sd'][num], height, 'standard deviation', color='black', ha='right', va='top', rotation=90)
        plt.axvline(x = az.summary(inf_data)['mean'][num] + az.summary(inf_data)['sd'][num], color = 'black')
        plt.text(az.summary(inf_data)['mean'][num] + az.summary(inf_data)['sd'][num], height, 'standard deviation', color='black', ha='right', va='top', rotation=90)

        plt.axvline(x = p3, color = 'black')
        plt.text(p3, height, '3%', color='black', ha='right', va='top', rotation=90)
        plt.axvline(x = p97, color = 'black')
        plt.text(p97, height, '97%', color='black', ha='right', va='top', rotation=90)
        plt.show()
        
def examine_chains(posterior, data_params, n_pendulums = 8, chain_length = 5000, n_chains = 4):
    chain_colors = ['#B5CA8D',
                    '#8BB174',
                    '#426B69',
                    '#222E50']
    plt.clf()
    for p in range(n_pendulums-1):


        for chain in range(n_chains-1):
            plt.plot(posterior['L'][chain_length * chain : chain_length * chain + chain_length-1 , p],
                     color = chain_colors[chain])
        plt.axhline(y = data_params['length'][p], color = 'black')
    plt.ylim([1,25])
    plt.title('Length inference')
    plt.show()

    fig, axs = plt.subplots(2, 4, figsize=(10, 10))

    for p, ax in enumerate(axs.flatten()):
        ax.axvline(x = data_params['length'][p], color = 'black')

        for chain in range(n_chains-1):
            if chain == 0:
                values, bins = np.histogram(posterior['L'][:, p],
                                             bins = 20)
                #print('values', values)
                #print('bins', bins)
            ax.hist(posterior['L'][chain_length * chain : chain_length * chain + chain_length-1 , p],
                    bins = bins,
                    color = chain_colors[chain], histtype='bar', ec='white',
                    density = True)
        #ax.set_xlim([1,25])


    plt.show()

    plt.clf()
    for p in range(n_pendulums-1):


        for chain in range(n_chains-1):
            plt.plot(posterior['theta'][chain_length * chain : chain_length * chain + chain_length-1 , p],
                     color = chain_colors[chain])
        plt.axhline(y = data_params['theta'][p], color = 'black')
    plt.ylim([0, np.pi/2])
    plt.title('theta inference')
    plt.show()

    fig, axs = plt.subplots(2, 4, figsize=(10, 10))

    for p, ax in enumerate(axs.flatten()):
        ax.axvline(x = data_params['theta'][p], color = 'black')
        for chain in range(n_chains-1):
            if chain == 0:
                values, bins = np.histogram(posterior['theta'][:, p],
                                             bins = 20)
                #print('values', values)
                #print('bins', bins)
            ax.hist(posterior['theta'][chain_length * chain : chain_length * chain + chain_length-1 , p],
                    bins = bins,
                    color = chain_colors[chain], histtype='bar', ec='white',
                    density = True)
        #ax.set_xlim([1,25])


    plt.show()

