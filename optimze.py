import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit
import vector
from deap import base, creator, tools, algorithms
import multiprocessing
from tqdm import tqdm


# Function to calculate the significance
def calculate_significance(individual, photon_pt, photon_eta, photon_phi, photon_E, photon_isTightID, photon_isolation, jet_pt):
    photon_pt_threshold_lead, photon_pt_threshold_sublead, jet_pt_threshold, photon_eta_threshold, photon_isolation_threshold = individual

    masses = []

    for event in range(len(photon_pt)):
        sorted_indices = np.argsort(photon_pt[event])[::-1]  # Sort photon transverse momenta in descending order

        if len(sorted_indices) < 2:
            continue

        # Photon pT Cuts
        if photon_pt[event][sorted_indices[0]] < photon_pt_threshold_lead or photon_pt[event][sorted_indices[1]] < photon_pt_threshold_sublead:
            continue

        # Jet Veto (Allow up to 1 jet)
        if sum(jet > jet_pt_threshold for jet in jet_pt[event]) > 1:
            continue

        # Photon ID and Isolation Cuts
        if not photon_isTightID[event][sorted_indices[0]] or not photon_isTightID[event][sorted_indices[1]]:
            continue

        if photon_isolation[event][sorted_indices[0]] > photon_isolation_threshold or photon_isolation[event][sorted_indices[1]] > photon_isolation_threshold:
            continue

        # Pseudorapidity (Eta) Cuts
        if abs(photon_eta[event][sorted_indices[0]]) > photon_eta_threshold or abs(photon_eta[event][sorted_indices[1]]) > photon_eta_threshold:
            continue

        # Create four-momentum vectors for the two photons using transverse momentum
        pt1, pt2 = photon_pt[event][sorted_indices[0]], photon_pt[event][sorted_indices[1]]
        eta1, eta2 = photon_eta[event][sorted_indices[0]], photon_eta[event][sorted_indices[1]]
        phi1, phi2 = photon_phi[event][sorted_indices[0]], photon_phi[event][sorted_indices[1]]
        E1, E2 = photon_E[event][sorted_indices[0]], photon_E[event][sorted_indices[1]]

        photon1 = vector.obj(px=pt1 * np.cos(phi1), py=pt1 * np.sin(phi1), pz=pt1 * np.sinh(eta1), E=E1)
        photon2 = vector.obj(px=pt2 * np.cos(phi2), py=pt2 * np.sin(phi2), pz=pt2 * np.sinh(eta2), E=E2)

        # Calculate the invariant mass
        mass = (photon1 + photon2).mass
        masses.append(mass)

    # Define histogram and background fit
    hist, bins = np.histogram(masses, bins=5000, range=(0, 200000))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit background in a sideband region (excluding the signal region)
    sideband_mask = ((bin_centers > 1e5) & (bin_centers < 1.2e5)) | ((bin_centers > 1.3e5) & (bin_centers < 1.6e5))
    popt, _ = curve_fit(poly_background, bin_centers[sideband_mask], hist[sideband_mask], p0=(3500, 1/100000))
    background = poly_background(bin_centers, *popt)

    # Signal subtraction
    signal = hist - background

    # Define signal region (115-135 GeV)
    signal_region_mask = (bin_centers > 1.15e5) & (bin_centers < 1.35e5)
    S = np.sum(signal[signal_region_mask])
    B = np.sum(background[signal_region_mask])

    # Calculate significance
    if B > 0:
        return S / np.sqrt(B),  # Comma to return a tuple (required by DEAP)
    else:
        return 0,


# Define an exponential function for the background fit
def poly_background(x, a, b):
    return a * np.exp(b * x)


# Load the ROOT file and data
def load_data():
    root_file = uproot.open("V:/Programme/Higgs/GamGam/MC/mc_343981.ggH125_gamgam.GamGam.root")
    tree = root_file["mini"]
    
    photon_pt = tree["photon_pt"].array(library="np")
    photon_eta = tree["photon_eta"].array(library="np")
    photon_phi = tree["photon_phi"].array(library="np")
    photon_E = tree["photon_E"].array(library="np")
    photon_isTightID = tree["photon_isTightID"].array(library="np")
    photon_isolation = tree["photon_etcone20"].array(library="np")
    jet_pt = tree["jet_pt"].array(library="np")
    
    return photon_pt, photon_eta, photon_phi, photon_E, photon_isTightID, photon_isolation, jet_pt


# Set up the DEAP genetic algorithm
def genetic_algorithm():
    # Load the data
    photon_pt, photon_eta, photon_phi, photon_E, photon_isTightID, photon_isolation, jet_pt = load_data()

    # Define parameter bounds
    param_bounds = [
        (30000, 60000),  # photon_pt_threshold_lead
        (20000, 40000),  # photon_pt_threshold_sublead
        (30000, 50000),  # jet_pt_threshold
        (1.8, 2.5),      # photon_eta_threshold
        (2000, 10000)    # photon_isolation_threshold
    ]

    # Set up DEAP toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Register gene function with random initialization within bounds
    for i, bounds in enumerate(param_bounds):
        toolbox.register(f"attr_{i}", np.random.uniform, bounds[0], bounds[1])

    # Register individual (chromosome) and population
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3, toolbox.attr_4), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation function
    toolbox.register("evaluate", calculate_significance, photon_pt=photon_pt, photon_eta=photon_eta, 
                     photon_phi=photon_phi, photon_E=photon_E, photon_isTightID=photon_isTightID, 
                     photon_isolation=photon_isolation, jet_pt=jet_pt)

    # Register genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", multiprocessing.Pool().map)

    # Set up progress bar
    pop_size = 100
    num_generations = 10
    pbar = tqdm(total=num_generations, desc="Evolution Progress")

    # Algorithm with genetic operations
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # To track the best individual

    # Define statistics for logging
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))
        hof.update(population)
        record = stats.compile(population)
        pbar.update(1)  # Update progress bar
        print(record)

    pbar.close()

    return hof[0]  # Return the best individual

if __name__ == '__main__':
    best_individual = genetic_algorithm()
    print(f"Best parameters: {best_individual}")
