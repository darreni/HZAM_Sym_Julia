# HZAM_Sym_Julia_V1.jl
# First public release of HZAM_Sym_Julia
# by Darren Irwin, begun in early July 2021
# Re-written from the R version of HZAM_Sym

# Please note that this version is not intended to be super user-friendly software 
# for people who aren't familiar with the Julia language. Rather, I am making this available
# so that those who know the basics of Julia can run this and tinker if they desire,
# and so that the community can see exactly what code produced the results in the paper.

# If you use this code, please cite the paper below, and possibly my GitHub or Dryad repository 
# where you obtained this code:

# Irwin, D., and Schluter, D. Hybridization and the coexistence of species. bioRxiv 2021.04.04.438369; doi: https://doi.org/10.1101/2021.04.04.438369 

# You need to make sure these packages below are added to your Julia environment.
# to add, type "]" to get the "pkg>" prompt, then type e.g. "add Distributions";
# or do this: 
# import Pkg; 
# Pkg.add("Distributions") 
# Pkg.add("Statistics") 
# Pkg.add("JLD2")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Plots")
# Pkg.add("CategoricalArrays")
# Pkg.add("Colors")
# Pkg.add("ColorSchemes")
# Pkg.add("CategoricalArrays")

using Distributions # needed for "Poisson" function
using Statistics  # needed for "mean" function
using JLD2 # needed for saving / loading data in Julia format
using CSV # for saving in csv format
using DataFrames # for converting data to save as csv

# for plotting:
using Plots
gr()  # use GR backend for graphs
using CategoricalArrays
using Colors, ColorSchemes
import ColorSchemes.plasma
using Plots.PlotMeasures  # needed for plot margin adjustment

# to start Julia with multiple threads, type in terminal e.g.:
# julia --threads 4
# To check, type in Julia: Threads.nthreads()
# 4

# set up functions (should not define functions repeatedly in loop, as causes re-compilation, slows things)

function generate_genotype_array(N_pop0,N_pop1,loci)
    total_N = N_pop0 + N_pop1  
    genotypes = Array{Int8, 3}(undef, 2, loci, total_N) # The "Int8" is the type (8-bit integer), and "undef" means an unitialized array, so values are meaningless
    genotypes[:,:,1:N_pop0] .= 0  # assigns genotypes of pop1
    genotypes[:,:,(N_pop0+1):total_N] .= 1  # assigns genotypes of pop
    return genotypes
end

function calc_traits_additive(genotypes)
    N = size(genotypes, 3) 
    traits = Array{Float32, 1}(undef, N) # Float32 should be enough precision; memory saving compared to Float64
    for i in 1:N
        traits[i] = mean(genotypes[:,:,i])
    end
    return traits
end

function get_survival_fitnesses_epistasis(genotypes, w_hyb, beta=1)
    survival_HI = calc_traits_additive(genotypes)
    epistasis_fitnesses = 1 .- (1 - w_hyb) .* (4 .* survival_HI .* (1 .- survival_HI)).^beta
    return epistasis_fitnesses 
end

function get_survival_fitnesses_hetdisadvantage(genotypes, w_hyb)
    N = size(genotypes, 3)
    num_loci = size(genotypes, 2)
    s_per_locus = 1 - w_hyb ^ (1/num_loci)  # loss in fitness due to each heterozygous locus 
    num_hetloci = Array{Int16, 1}(undef, N)
    for ind in 1:N  # count number of het loci per individual
        num_hetloci[ind] = sum(genotypes[1,:,ind] .!= genotypes[2,:,ind])
    end
    hetdisadvantage_fitnesses = (1 - s_per_locus) .^ num_hetloci
    return hetdisadvantage_fitnesses
end 

function run_HZAM(set_name::String, ecolDiff, intrinsic_R, replications;  # the semicolon makes the following optional keyword arguments 
    K_total::Int = 1000, max_generations::Int = 1000, 
    total_loci::Int = 6, female_mating_trait_loci = 1:3, male_mating_trait_loci = 1:3,
    competition_trait_loci = 1:3, hybrid_survival_loci = 1:3, neutral_loci = 4:6,
    survival_fitness_method::String = "epistasis", per_reject_cost = 0,
    starting_pop_ratio = 1.0)
    # ecolDiff should be from 0 to 1 (parameter "E" in the paper)
    # intrinsic_R is called "R" in the paper
    # replications should be somehting like "1:10" or just "1" for 1 replicate, or something like "2:5" to add replicates after 1 is done
    # K_total (default 1000) is the total carrying capacity, divided equally between K_alpha and K_beta
    # max_generations (default 1000) is the number of generations each simulation will run 
    # total_loci (default 6) is the total number of loci, regardless of their role (with indices 1:total_loci, referred to below)
    # Need to specify indices (columns) of four types of functional loci (can be the same). At least one should begin with index 1:
    # female_mating_trait_loci (default 1:3) is indices of the loci that determine the female mating trait
    # male_mating_trait_loci (default 1:3) is indices of the loci that determine the male mating trait
    # competition_trait_loci (default 1:3) is indices of the loci that determine the ecological trait (used in fitness related to resource use)
    # hybrid_survival_loci (default 1:3) is indices of the loci that determine survival probability of offspring to adulthood (can be viewed as incompatibilities and/or fitness valley based on ecology)
    # Specify indices (columns) of neutral loci (which have no effect on anything, just along for the ride):
    # neutral_loci (default 4:6) is indices of neutral loci (used for neutral measure of hybrid index; not used in the HZAM-sym paper)
    # per_reject_cost (default 0, can take values of 0 to 1) is fitness loss of female per male rejected (due to search time, etc.)
    # starting_pop_ratio (default 1) is the starting ratio of pop A to pop B

    save_outcomes_JL = true
    save_outcomes_csv = true  # whether to save the whole outcome array as csv files (with each rep as separate file)
    save_each_sim = false  # whether to save detailed data for each simulation

    # the set of hybrid fitnesses (w_hyb) values that will be run
    w_hyb_set = [1, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0] # for just one run, just put one number in this and next line
    # the set of assortative mating strengths (S_AM) that will be run
    S_AM_set = [1, 3, 10, 30, 100, 300, 1000, Inf]  # ratio of: probably of accepting homospecific vs. prob of accepting heterospecific

    # get the chosen survival fitness function
    if survival_fitness_method == "epistasis"
        get_survival_fitnesses = get_survival_fitnesses_epistasis
        short_survFitnessMethod = "Ep"
    elseif survival_fitness_method == "hetdisadvantage"
        get_survival_fitnesses = get_survival_fitnesses_hetdisadvantage
        short_survFitnessMethod = "Het"
    else
        println("ERROR--no survival fitness method chosen--should be either epistasis or hetdisadvantage")
    end

    total_functional_loci = max(maximum(female_mating_trait_loci), maximum(male_mating_trait_loci), maximum(competition_trait_loci), maximum(hybrid_survival_loci))
    functional_loci_range = 1:total_functional_loci
    num_neutral_loci = length(Vector(neutral_loci))
    if total_functional_loci + num_neutral_loci ≠ total_loci
        println("#### WARNING: Please examine your loci numbers and indices, as they don't all match up ####")
    end 

    # intrinsic_R = 1.05  # Intrinsic growth rate, this is the average maximum expected number of offspring per individual, when pop size far below K
    K_A = K_total / 2  # EVEN NUMBER; carrying capacity (on resource alpha) of entire range (for two sexes combined), regardless of species 
    K_B = K_total / 2   # EVEN NUMBER; carrying capacity (on resource beta) of entire range (for two sexes combined), regardless of species
    
    pop0_starting_N = K_A   # starting N of species 0
    pop0_starting_N_half = Int(pop0_starting_N/2)
    pop1_starting_N = Int(round(starting_pop_ratio * K_B))   # starting N of species 1, which can be lower if starting_pop_ratio is below 1)
    pop1_starting_N_half = Int(pop1_starting_N/2)

    beta = 1  # the epistasis parameter beta

    # specify ecological resource competitive abilities for two resources A and B 
    # ecolDiff = 1.0 # this is "E" in the paper 
    competAbility_useResourceA_species0 = (1 + ecolDiff)/2    # equals 1 when ecolDiff = 1   
    competAbility_useResourceB_species0 = 1 - competAbility_useResourceA_species0
    competAbility_useResourceA_species1 = (1 - ecolDiff)/2   # equals 0 when ecolDiff = 1
    competAbility_useResourceB_species1 = 1 - competAbility_useResourceA_species1

    # set up array of strings to record outcomes
    outcome_array = Array{String, 3}(undef, length(w_hyb_set), length(S_AM_set), length(replications))

    for k in 1:length(replications)  # loop through the replicate runs
        replicate_ID = replications[k]

        run_set_name = string(set_name,"_rep", replicate_ID)

        # Loop through the different simulation sets
        Threads.@threads for i in 1:length(w_hyb_set)
            for j in 1:length(S_AM_set) 
                w_hyb = w_hyb_set[i]
                S_AM = S_AM_set[j]
                println("w_hyb = ",w_hyb,"; S_AM = ",S_AM)

                run_name = string("HZAM_animation_run",run_set_name,"_surv",short_survFitnessMethod,"_ecolDiff",ecolDiff,"_growthrate",intrinsic_R,"_K",K_total,"_FL",total_functional_loci,"_NL",num_neutral_loci,"_gen",max_generations,"_SC",per_reject_cost,"_Whyb",w_hyb,"_SAM",S_AM)
                
                # convert S_AM to pref_ratio (for use in math below)
                if S_AM == 1
                    S_AM_for_math = 1 + 10^(-15)
                elseif S_AM == Inf
                    S_AM_for_math = 10^(15)
                else 
                    S_AM_for_math = S_AM
                end
                pref_SD = sqrt( -1 / (2 * log(1/S_AM_for_math)))  # width of female acceptance curve for male trait

                # set up initial values for one simulation
                extinction = false
                outcome = []
                final_distribution = []
                # Generate genotype array for population of females:
                # this is a 3D array, where rows (D1) are alleles (row 1 from mother, row 2 from father),
                # columns (D2) are loci, and pages (D3) are individuals
                genotypes_F = generate_genotype_array(pop0_starting_N_half, pop1_starting_N_half, total_loci)
                genotypes_M = generate_genotype_array(pop0_starting_N_half, pop1_starting_N_half, total_loci)
                # functional loci are first, followed by neutral loci

                for generation in 1:max_generations
                    
                    # Prepare for mating and reproduction
                    N_F = size(genotypes_F, 3)
                    N_M = size(genotypes_M, 3)
                    
                    # println("generation: ", generation, "; individuals: ", N_F + N_M)

                    # calculate mating trait values (T) from genotypes
                    female_mating_traits = calc_traits_additive(genotypes_F[:,female_mating_trait_loci,:])
                    male_mating_traits = calc_traits_additive(genotypes_M[:,male_mating_trait_loci,:])

                    # calculate ecological competition trait values from genotypes
                    competition_traits_F = calc_traits_additive(genotypes_F[:,competition_trait_loci,:])
                    competition_traits_M = calc_traits_additive(genotypes_M[:,competition_trait_loci,:])

                    # calculate individual contributions to resource use, according to linear gradient between use of species 0 and species 1
                    ind_useResourceA_F = competAbility_useResourceA_species1 .+ ((1 .- competition_traits_F) .* (competAbility_useResourceA_species0 - competAbility_useResourceA_species1))
                    ind_useResourceB_F = competAbility_useResourceB_species0 .+ (competition_traits_F .* (competAbility_useResourceB_species1 - competAbility_useResourceB_species0))
                    ind_useResourceA_M = competAbility_useResourceA_species1 .+ ((1 .- competition_traits_M) .* (competAbility_useResourceA_species0 - competAbility_useResourceA_species1))
                    ind_useResourceB_M = competAbility_useResourceB_species0 .+ (competition_traits_M .* (competAbility_useResourceB_species1 - competAbility_useResourceB_species0))
                    # sum up the resource use over all individuals
                    total_useResourceA = sum(ind_useResourceA_F) + sum(ind_useResourceA_M)
                    total_useResourceB = sum(ind_useResourceB_F) + sum(ind_useResourceB_M)
                    # calculate growth rates due to each resource (according to discrete time logistic growth equation)
                    growth_rate_ResourceA = (intrinsic_R * K_A) / (K_A + ((total_useResourceA)*(intrinsic_R - 1)))
                    growth_rate_ResourceB = (intrinsic_R * K_B) / (K_B + ((total_useResourceB)*(intrinsic_R - 1)))

                    # Set up structure to record number of matings per male (and female, although almost always 1 for females), 
                    # to determine sexual selection due to HI class:
                    matings_per_male = zeros(Int8, N_M)
                    matings_per_female = zeros(Int8, N_F)

                    # create empty data structures for keeping track of numbers offspring of parents
                    daughters_per_mother = zeros(Int16, N_F) 
                    sons_per_mother = zeros(Int16, N_F) 
                    daughters_per_father = zeros(Int16, N_M)  
                    sons_per_father = zeros(Int16, N_M)  

                    # make empty arrays for storing genotypes of daughters and sons
                    genotypes_daughters = Array{Int8, 3}(undef, 2, total_loci, 0) 
                    genotypes_sons = Array{Int8, 3}(undef, 2, total_loci, 0)

                    # create structures for recording indices of mother and father (for error checking, and potentially for tracking genealogies)
                    # first column for mother index (3rd dim of genotypes_F) and second column for father index (3rd dim of genotypes_M) 
                    daughter_parent_IDs = Array{Int}(undef, 0, 2)
                    son_parent_IDs = Array{Int}(undef, 0, 2) 

                    # loop through mothers, mating and reproducing
                    for mother in 1:N_F
                        # initialize tracking variables
                        mate = false # becomes true when female is paired with male
                        rejects = 0 # will track number of rejected males (in cases there is cost--which there isn't in main HZAM-Sym paper)
                        father = [] # will contain the index of the male mate
                        # make vector of indices of eligible males
                        elig_M = Vector(1:N_M)
                        # determine male mate of female
                        while mate == false
                            # present female with random male, and remove him from list using "splice!" 
                            focal_male = splice!(elig_M, rand(eachindex(elig_M)))
                            # compare male trait with female's trait (preference), and determine
                            # whether she accepts; note that match_strength is determined by a
                            # Gaussian, with a maximum of 1 and minimum of zero.
                            match_strength = (exp(1) ^ ((-(male_mating_traits[focal_male] - female_mating_traits[mother])^2) / (2 * (pref_SD ^2))))
                            if rand() < match_strength
                                # she accepts male, and mates
                                father = focal_male
                                matings_per_male[focal_male] += 1 # this adds 1 to the matings for that male
                                matings_per_female[mother] += 1 
                                mate = true
                            else
                                # she rejects male
                                rejects += 1
                                if length(elig_M) == 0
                                    break
                                end
                            end
                        end
                        # determine fitness cost due to mate search (number of rejected males)
                        search_fitness = (1-per_reject_cost) ^ rejects    # (in most of HZAM-Sym paper, per_reject_cost = 0)

                        # determine fitness due to female use of available resources
                        growth_rate_of_focal_female = (ind_useResourceA_F[mother] * growth_rate_ResourceA) + (ind_useResourceB_F[mother] * growth_rate_ResourceB)
                        #combine for total fitness:   
                        reproductive_fitness = 2 * growth_rate_of_focal_female * search_fitness  # the 2 is because only females, not males, produce offspring
                        # now draw the number of offspring from a poisson distribution with a mean of reproductive_fitness
                        if isempty(father)
                            offspring = 0  # if no mate (because all males were rejected), then no offspring
                        else
                            offspring = rand(Poisson(reproductive_fitness)) 
                        end
                        # if offspring, generate their genotypes and sexes
                        if offspring >= 1
                            for kid in 1:offspring
                                kid_info = Array{Int8, 2}(undef, 2, total_loci) # place to store genotype of one offspring
                                # generate genotypes; for each locus (column), first row for allele from mother, second row for allele from father
                                for locus in 1:total_loci
                                    kid_info[1,locus] = genotypes_F[rand([1 2]), locus, mother]  # for this locus, pick a random allele from the mother
                                    kid_info[2,locus] = genotypes_M[rand([1 2]), locus, father] 
                                end
                                # determine sex of kid and add to table
                                if rand() > 0.5 # kid is daughter
                                    genotypes_daughters = cat(genotypes_daughters, kid_info, dims=3)
                                    daughter_parent_IDs = cat(daughter_parent_IDs, [mother father], dims=1)
                                    daughters_per_mother[mother] += 1
                                    daughters_per_father[father] += 1 
                                else # kid is son
                                    genotypes_sons = cat(genotypes_sons, kid_info, dims=3)
                                    son_parent_IDs = cat(son_parent_IDs, [mother father], dims=1)
                                    sons_per_mother[mother] += 1
                                    sons_per_father[father] += 1 
                                end
                            end
                        end
                    end # of loop through mothers

                    # check if either no daughters or no sons, and end the simulation if so
                    if (size(genotypes_daughters, 3) == 0) || (size(genotypes_sons, 3) == 0)
                        extinction = true # record an extinction of whole population (both "species")
                        break # break out of current loop (this simulation) 
                    end
                    
                    # For someday: add in here the option of tracking fitness?
                    
                    # determine survival fitnesses of daughters due to epistasis

                    survival_fitness_daughters = get_survival_fitnesses(genotypes_daughters[:,hybrid_survival_loci,:], w_hyb)
                    daughters_survive = survival_fitness_daughters .> rand(length(survival_fitness_daughters))
                    # same for sons:
                    survival_fitness_sons = get_survival_fitnesses(genotypes_sons[:,hybrid_survival_loci,:], w_hyb)
                    sons_survive = survival_fitness_sons .> rand(length(survival_fitness_sons))

                    # check if either no surviving daughters or no surviving sons, and end the simulation if so
                    if (sum(daughters_survive) == 0) || (sum(sons_survive) == 0)
                        extinction = true # record an extinction of whole population (both "species")
                        break # break out of current loop (this simulation) 
                    end
                
                    # assign surviving offspring to new adult population
                    genotypes_F = genotypes_daughters[:,:,daughters_survive]
                    genotypes_M = genotypes_sons[:,:,sons_survive] 

                end # of loop through generations
                    
                # Record results of the one simulation
                functional_HI_all_inds = []
                species0_proportion = []
                species1_proportion = []
                HI_NL_all_inds = []
                species0_proportion_NL = []
                species1_proportion_NL = [] 
                if extinction  # whole simulation went extinct
                    outcome = "extinction"
                    if save_each_sim
                        @save string("simulation_data.",run_name,".jld2") outcome
                    end
                else  # no complete extinction
                    # use trait loci to calculate HI of each individual
                    functional_HI_all_inds = [calc_traits_additive(genotypes_F[:,functional_loci_range,:]); calc_traits_additive(genotypes_M[:,functional_loci_range,:])]
                    # calculate proportion of all individuals who are species0 or species1 (defined as low and high 10% of HI distribution, respectively)
                    species0_proportion = sum(functional_HI_all_inds .< 0.1) / length(functional_HI_all_inds)
                    species1_proportion = sum(functional_HI_all_inds .> 0.9) / length(functional_HI_all_inds)
                    if species0_proportion >= 0.85 || species1_proportion >= 0.85
                        outcome = "one_species"
                    elseif (species0_proportion + species1_proportion >= 0.85) && (species0_proportion >= 0.15) && (species1_proportion >= 0.15)
                        outcome = "two_species"
                    else
                        outcome = "blended"
                    end
                    HI_NL_all_inds = [calc_traits_additive(genotypes_F[:,neutral_loci,:]); calc_traits_additive(genotypes_M[:,neutral_loci,:])]
                    species0_proportion_NL = sum(HI_NL_all_inds .== 0) / length(HI_NL_all_inds)
                    species1_proportion_NL = sum(HI_NL_all_inds .== 1) / length(HI_NL_all_inds)
                    if save_each_sim
                        @save string("HZAM_Sym_Julia_results_GitIgnore/simulation_data.",run_name,".jld2") outcome functional_HI_all_inds species0_proportion species1_proportion HI_NL_all_inds species0_proportion_NL species1_proportion_NL
                    end 
                end
                println(run_name, "  outcome was: ", outcome)
                outcome_array[i,j,k] = outcome
            end # of S_AM loop
        end # of w_hyb loop   
    end # of replicate loop

    if save_outcomes_JL
        filename = string("HZAM_Sym_Julia_results_GitIgnore/outcomeArray_set",set_name,"_surv",short_survFitnessMethod,"_ecolDiff",ecolDiff,"_growthrate",intrinsic_R,"_K",K_total,"_FL",total_functional_loci,"_NL",num_neutral_loci,"_gen",max_generations,"_SC",per_reject_cost,".jld2")
        save_object(filename, outcome_array)
    end
 
    if save_outcomes_csv
        for i in 1:size(outcome_array, 3)
            filename = string("HZAM_Sym_Julia_results_GitIgnore/outcomeArray_set",set_name,"_surv",short_survFitnessMethod,"_ecolDiff",ecolDiff,"_growthrate",intrinsic_R,"_K",K_total,"_FL",total_functional_loci,"_NL",num_neutral_loci,"_gen",max_generations,"_SC",per_reject_cost,"_rep",replications[i])
            CSV.write(filename, Tables.table(outcome_array[:,:,i]), writeheader=false)
        end 
    end
    return outcome_array
end 

#### functions for summarizing and plotting results

# convert outcome array (an array of strings) to categorical array
function convert_to_cat_array(outcome_array)
    cat_outcome_array = compress(CategoricalArray(outcome_array))
    levels!(cat_outcome_array, ["extinction", "blended", "one_species", "two_species"])
    return cat_outcome_array
end 

# function for plotting grid of pie charts showing distribution of four outcomes, using categorical array as input
function plot_all_outcomes(cat_outcome_array)
    num_outcome_types = length(levels(cat_outcome_array))
    outcome_counts = Array{Int16, 2}(undef, num_outcome_types, (size(cat_outcome_array, 1)*size(cat_outcome_array, 2) )) 
    for i in 1:size(cat_outcome_array, 1) 
        for j in 1:size(cat_outcome_array, 2) 
            for outcome_num in 1:num_outcome_types
                outcome_counts[outcome_num, j + (i-1)*size(cat_outcome_array, 2)] = sum(cat_outcome_array[i,j,:] .== levels(cat_outcome_array)[outcome_num])
            end
        end
    end
    colors_of_outcomes = [RGB(0,0,0), plasma[0.525], plasma[0.2], plasma[0.9]] # colors for 4 outcome categories
    pie(outcome_counts, layout = grid(size(cat_outcome_array, 1), size(cat_outcome_array, 2)), legend = false, palette = colors_of_outcomes, margin = -2.0mm)
    plot!(size=(800,1300))
end

function get_most_common_outcomes(cat_outcome_array)
    levels_of_outcomes = levels(cat_outcome_array)
    num_outcome_types = length(levels_of_outcomes)
    most_common_outcomes = CategoricalArray{String, 2}(undef, size(cat_outcome_array, 1), size(cat_outcome_array, 2))
    levels!(most_common_outcomes, ["extinction", "blended", "one_species", "two_species"])
    for i in 1:size(cat_outcome_array, 1) 
        for j in 1:size(cat_outcome_array, 2)
            outcome_counts = Vector{Int}(undef, num_outcome_types) 
            for outcome_num in 1:num_outcome_types
                outcome_counts[outcome_num] = sum(cat_outcome_array[i,j,:] .== levels(cat_outcome_array)[outcome_num])
            end
            outcomes_with_max_count = findall(outcome_counts .== maximum(outcome_counts))
            if length(outcomes_with_max_count) == 1
                most_common_outcomes[i,j] = levels_of_outcomes[outcomes_with_max_count[1]]
            elseif length(outcomes_with_max_count) >= 2
                # if tie in outcome count, choose one randomly
                most_common_outcomes[i,j] = levels_of_outcomes[sample(outcomes_with_max_count)]
            end
        end
    end
    return most_common_outcomes
end

function plot_common_outcomes(common_outcome_array)
    # make heat map of outcomes
    colors_of_outcomes = [RGB(0,0,0), plasma[0.525], plasma[0.2], plasma[0.9]] # colors for 4 outcome categories
    one_outcome_array = reverse(common_outcome_array, dims=1)
    x_midpoints = log10.([1, 3, 10, 30, 100, 300, 1000, 5000])  # the S_AM values, with Inf convert to 3000 for graphing 
    w_hyb_set = [1, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0] # for just one run, just put one number in this and next line
    y_midpoints = reverse(w_hyb_set)
    min_color = minimum(levelcode.(one_outcome_array)) # this and next line needed to choose the proper colors for the figure
    max_color = maximum(levelcode.(one_outcome_array))
    heatmap(x_midpoints, y_midpoints, one_outcome_array, c = colors_of_outcomes[min_color:max_color], yflip = false, tick_direction = :out, colorbar = false, size = (440,310), framestyle = :box)
    xaxis!("Strength of conspecific mate preference")
    xticklabels = ["1", "3", "10", "30", "100", "300", "1000", "complete"]
    plot!(xticks=(x_midpoints, xticklabels))
    yaxis!("Hybrid fitness")
    yticklabels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "", "", "1.0"]
    plot!(yticks=(y_midpoints, yticklabels), tick_direction = :out)
    # add white lines
    plot!([xlims()[1], xlims()[2]], [0.05, 0.05], linecolor = :white, widen = false, legend = false, linewidth=3)
    x_for_line = mean(x_midpoints[[length(x_midpoints)-1 length(x_midpoints)]])
    plot!([x_for_line, x_for_line], [ylims()[1], ylims()[2]], linecolor = :white, widen = false, legend = false, linewidth=3)
end

function make_and_save_figs(ResultsFolder, RunName, RunOutcomes)
    cat_RunOutcomes = convert_to_cat_array(RunOutcomes)
    display(plot_all_outcomes(cat_RunOutcomes))
    savefig(string(ResultsFolder,"/",RunName,"_AllOutcomes.png"))
    savefig(string(ResultsFolder,"/",RunName,"_AllOutcomes.pdf"))
    most_common_outcomes = get_most_common_outcomes(cat_RunOutcomes)
    display(plot_common_outcomes(most_common_outcomes))
    savefig(string(ResultsFolder,"/",RunName,"_MostCommonOutcomes.png"))
    savefig(string(ResultsFolder,"/",RunName,"_MostCommonOutcomes.pdf"))
end

#### Run the actual simulation by calling the above functions:

# If wanting to run a test to make sure things are working (after above functions are loaded into memory):
# RunOutcomes = run_HZAM("TEST", 1.0, 1.05, 1:3, 100, 50)

ResultsFolder = "HZAM_Sym_Julia_results_GitIgnore" # or change to another folder where results will be saved

# To run simulations under the parameters in the paper, you can execute the following,
# under the heading for each Figure panel in the paper. Because these are stochastic simulations,
# results will differ a little bit (but not much) from those shown in the paper.
# Also, the "make_and_save_figs" function has saved the figure panels under each set of conditions run below,
# and the main script saved the Julia object where the results are stored--this starts with "outcomeArray_set"
# and will be available on Dryad.

# (Note that if the below have "Het" in their name, they were done using the heterozygote disadvantage (underdominance) 
# option for survival fitness. If they do not have "Het" in their name, they were done using the epistasis option.)


# Figure 3A
RunName = "JL_fig3aHet"
RunOutcomes = run_HZAM(RunName, 0.0, 1.05, 1:25;
    survival_fitness_method = "hetdisadvantage")
# took about 5 hours to run, with 4 threads on a Mac desktop with 4 cores
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 3B
RunName = "JL_fig3bHet"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Figure 4A
RunName = "JL_Fig4aHet"
RunOutcomes = run_HZAM(RunName, 1.0, 1.025, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 4B
RunName = "JL_Fig4bHet"
RunOutcomes = run_HZAM(RunName, 1.0, 1.2, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 4C
RunName = "JL_Fig4cHet"
RunOutcomes = run_HZAM(RunName, 1.0, 2.6, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Figure 6 row 1 panel 2
RunName = "JL_Fig6(1,2)Het"
RunOutcomes = run_HZAM(RunName, 0.25, 1.05, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 6 row 1 panel 3
RunName = "JL_Fig6(1,3)Het"
RunOutcomes = run_HZAM(RunName, 0.5, 1.05, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 6 row 1 panel 4
RunName = "JL_Fig6(1,4)Het"
RunOutcomes = run_HZAM(RunName, 0.75, 1.05, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 6 row 2 panel 1
RunName = "JL_Fig6(2,1)Het"
RunOutcomes = run_HZAM(RunName, 0.0, 2.6, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 6 row 2 panel 2
RunName = "JL_Fig6(2,2)Het"
RunOutcomes = run_HZAM(RunName, 0.25, 2.6, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 6 row 2 panel 3
RunName = "JL_Fig6(2,3)Het"
RunOutcomes = run_HZAM(RunName, 0.5, 2.6, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure 6 row 2 panel 4
RunName = "JL_Fig6(2,4)Het"
RunOutcomes = run_HZAM(RunName, 0.75, 2.6, 1:25;
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Figure S2A (epistasis):
RunName = "JL_fig3a"
RunOutcomes = run_HZAM(RunName, 0.0, 1.05, 1:25)
# this run (which does 2600 simulations) takes about 4 hours (with 4 threads on a somewhat powerful desktop Mac)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S2B (epistasis):
RunName = "JL_fig3b"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Figure S7A
RunName = "JL_fig3bHet125gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    max_generations = 125, survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S7B
RunName = "JL_fig3bHet250gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    max_generations = 250, survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S7C
RunName = "JL_fig3bHet500gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    max_generations = 500, survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S7E
RunName = "JL_fig3bHet2000gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    max_generations = 2000, survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S7F
RunName = "JL_fig3bHet4000gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    max_generations = 4000, survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Figure S8A
RunName = "JL_LikeFig3bHet_butFL1"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    K_total = 1000, max_generations = 1000,
    total_loci = 2, female_mating_trait_loci = 1:1, male_mating_trait_loci = 1:1,
    competition_trait_loci = 1:1, hybrid_survival_loci = 1:1, neutral_loci = 2:2,
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S8C
RunName = "JL_LikeFig3bHet_butFL9"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    K_total = 1000, max_generations = 1000,
    total_loci = 18, female_mating_trait_loci = 1:9, male_mating_trait_loci = 1:9,
    competition_trait_loci = 1:9, hybrid_survival_loci = 1:9, neutral_loci = 10:18,
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S8D
RunName = "JL_LikeFig3bHet_butFL27"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    K_total = 1000, max_generations = 1000,
    total_loci = 54, female_mating_trait_loci = 1:27, male_mating_trait_loci = 1:27,
    competition_trait_loci = 1:27, hybrid_survival_loci = 1:27, neutral_loci = 28:54,
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Figure S9A
RunName = "JL_fig3bHetDiffLociTraitPref"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    K_total = 1000, max_generations = 1000,
    total_loci = 9, female_mating_trait_loci = 1:3, male_mating_trait_loci = 4:6,
    competition_trait_loci = 1:6, hybrid_survival_loci = 1:6, neutral_loci = 7:9,
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S9B
RunName = "JL_fig3bHetDiffLociTraitPrefEcol"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    K_total = 1000, max_generations = 1000,
    total_loci = 12, female_mating_trait_loci = 1:3, male_mating_trait_loci = 4:6,
    competition_trait_loci = 7:9, hybrid_survival_loci = 7:9, neutral_loci = 10:12,
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Figure S10C
RunName = "JL_LikeFig3bHet_but_starting_pop_ratio_0.5"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    survival_fitness_method = "hetdisadvantage",
    starting_pop_ratio = 0.5)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S10D
RunName = "JL_LikeFig4bHet_but_starting_pop_ratio_0.5"
RunOutcomes = run_HZAM(RunName, 1.0, 1.2, 1:25;
    survival_fitness_method = "hetdisadvantage",
    starting_pop_ratio = 0.5)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S10E
RunName = "JL_LikeFig3bHet_but_starting_pop_ratio_0.1"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25;
    survival_fitness_method = "hetdisadvantage",
    starting_pop_ratio = 0.1)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Figure S10F
RunName = "JL_LikeFig4bHet_but_starting_pop_ratio_0.1"
RunOutcomes = run_HZAM(RunName, 1.0, 1.2, 1:25;
    survival_fitness_method = "hetdisadvantage",
    starting_pop_ratio = 0.1)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# Not shown in paper (but available in saved results):
RunName = "JL_LikeFig3bHet_butFL2"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    K_total = 1000, max_generations = 1000,
    total_loci = 4, female_mating_trait_loci = 1:2, male_mating_trait_loci = 1:2,
    competition_trait_loci = 1:2, hybrid_survival_loci = 1:2, neutral_loci = 3:4,
    survival_fitness_method = "hetdisadvantage")
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_LikeFig3b_butFL1"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25, 
    1000, 1000,
    2, 1:1, 1:1,
    1:1, 1:1, 2:2)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_LikeFig3b_butFL9"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25, 
    1000, 1000,
    18, 1:9, 1:9,
    1:9, 1:9, 10:18)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_LikeFig3b_butFL27"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25, 
    1000, 1000,
    54, 1:27, 1:27,
    1:27, 1:27, 28:54)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_LikeFig4c_butFL27"
RunOutcomes = run_HZAM(RunName, 1.0, 2.6, 1:25, 
    1000, 1000,
    54, 1:27, 1:27,
    1:27, 1:27, 28:54)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig4a"
RunOutcomes = run_HZAM(RunName, 1.0, 1.025, 1:25) 
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig4b"
RunOutcomes = run_HZAM(RunName, 1.0, 1.2, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig4c"
RunOutcomes = run_HZAM(RunName, 1.0, 2.6, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig6(1,2)"
RunOutcomes = run_HZAM(RunName, 0.25, 1.05, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig6(1,3)"
RunOutcomes = run_HZAM(RunName, 0.5, 1.05, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig6(1,4)"
RunOutcomes = run_HZAM(RunName, 0.75, 1.05, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig6(2,1)"
RunOutcomes = run_HZAM(RunName, 0.0, 2.6, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig6(2,2)"
RunOutcomes = run_HZAM(RunName, 0.25, 2.6, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig6(2,3)"
RunOutcomes = run_HZAM(RunName, 0.5, 2.6, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_Fig6(2,4)"
RunOutcomes = run_HZAM(RunName, 0.75, 2.6, 1:25)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_fig3b125gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    1000, 125)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_fig3b250gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    1000, 250)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_fig3b500gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    1000,500)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_fig3b2000gen"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    1000,2000)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_fig3bDiffLociTraitPref"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    1000, 1000,
    9, 1:3, 4:6,
    1:6, 1:6, 7:9)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)

# Not shown in paper (but available in saved results), with epistasis:
RunName = "JL_fig3bDiffLociTraitPrefEcol"
RunOutcomes = run_HZAM(RunName, 1.0, 1.05, 1:25,
    1000, 1000,
    12, 1:3, 4:6,
    7:9, 7:9, 10:12)
make_and_save_figs(ResultsFolder, RunName, RunOutcomes)


# make graph of epistasis fitness (for Figure 1Cii):
survival_HI = collect(0.0:0.01:1.0)
beta = 1
w_hyb = 0.8
epistasis_fitnesses = 1 .- (1 - w_hyb) .* (4 .* survival_HI .* (1 .- survival_HI)).^beta
using Plots
display(plot(survival_HI, epistasis_fitnesses, ylims = (0,1)))
savefig("plot.pdf")
