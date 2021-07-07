# HZAM_Sym_Julia_beta1.jl
# by Darren Irwin, beginning in early July 2021
# Adapted from the R version of HZAM_Sym

# Here is a small test addition (to see if GitHub works)
# A second small test line

# Need to make sure these packages are added:
# Distributions
# to add, type "]" to get the "pkg>" prompt, then type e.g. "add Distributions" 
using Distributions # needed for "Poisson" function
using Statistics  # needed for "mean" function
using JLD2 # needed for saving / loading data in Julia format
using CSV # for saving in csv format
using DataFrames # for converting data to save as csv

# to start Julia with multiple threads, type in terminal e.g.:
# julia --threads 4
# To check, type in Julia: Threads.nthreads()
# 4
# (I tried more but the REPL doesn't seem to like more)

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

set_name = "JL1"

replications = 1 # collect(3)  # or just 1 for 1 replicate, or something like (2:5) to add replicates after 1 is done

# the set of hybrid fitnesses (w_hyb) values that will be run
w_hyb_set = [1, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0] # for just one run, just put one number in this and next line
# the set of assortative mating strengths (S_AM) that will be run
S_AM_set = [1, 3, 10, 30, 100, 300, 1000, Inf]  # ratio of: probably of accepting homospecific vs. prob of accepting heterospecific

trait_loci = 3  # these loci determine the mating trait and the HI for low hybrid fitness
neutral_loci = 3   # number of neutral loci (used for neutral measure of hybrid index; not used in the HZAM-sym paper)
total_loci = trait_loci + neutral_loci
trait_loci_cols = 1:trait_loci  # the column index range for loci determining the trait
neutral_loci_cols = (trait_loci+1 : trait_loci+neutral_loci) 


per_reject_cost = 0 # fitness loss of female per male rejected (due to search time, etc.)

intrinsic_R = 2.6  # Intrinsic growth rate, this is the average maximum expected number of offspring per individual, when pop size far below K
K_A = 500  # EVEN NUMBER; carrying capacity (on resource alpha) of entire range (for two sexes combined), regardless of species 
K_B = 500  # EVEN NUMBER; carrying capacity (on resource beta) of entire range (for two sexes combined), regardless of species
K_total = K_A + K_B
pop0_starting_N = K_A   # starting N of species 0
pop0_starting_N_half = Int(pop0_starting_N/2)
pop1_starting_N = K_B   # starting N of species 1
pop1_starting_N_half = Int(pop1_starting_N/2)

beta = 1  # the epistasis parameter beta

# specify ecological resource competitive abilities for two resources A and B 
ecolDiff = 1.0 # this is "E" in the paper 
competAbility_useResourceA_species0 = (1 + ecolDiff)/2    # equals 1 when ecolDiff = 1   
competAbility_useResourceB_species0 = 1 - competAbility_useResourceA_species0
competAbility_useResourceA_species1 = (1 - ecolDiff)/2   # equals 0 when ecolDiff = 1
competAbility_useResourceB_species1 = 1 - competAbility_useResourceA_species1

max_generations = 1000

# set up array of strings to record outcomes
outcome_array = Array{String, 3}(undef, length(w_hyb_set), length(S_AM_set), length(replications))

for k in 1:length(replications)  # loop through the replicate runs
    replicate_ID = replications[k]

    run_set_name = string(set_name,"_rep", replicate_ID)

    # Loop through the different simulation sets
    for i in 1:length(w_hyb_set)
        Threads.@threads for j in 1:length(S_AM_set) 
            w_hyb = w_hyb_set[i]
            S_AM = S_AM_set[j]
            println("w_hyb = ",w_hyb)
            println("S_AM = ",S_AM)

            run_name = string("HZAM_animation_run",run_set_name,"_ecolDiff",ecolDiff,"_growthrate",intrinsic_R,"_K",K_total,"_TL",trait_loci,"_gen",max_generations,"_Whyb",w_hyb,"_SAM",S_AM)
            
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
            # trait loci are first, followed by neutral loci

            for generation in 1:max_generations
                
                # Prepare for mating and reproduction
                N_F = size(genotypes_F, 3)
                N_M = size(genotypes_M, 3)
                
                # println("generation: ", generation, "; individuals: ", N_F + N_M)

                # calculate trait values (T) from genotypes
                traits_F = calc_traits_additive(genotypes_F[:,1:trait_loci,:])
                traits_M = calc_traits_additive(genotypes_M[:,1:trait_loci,:])

                # calculate individual contributions to resource use, according to linear gradient between use of species 0 and species 1
                ind_useResourceA_F = competAbility_useResourceA_species1 .+ ((1 .- traits_F) .* (competAbility_useResourceA_species0 - competAbility_useResourceA_species1))
                ind_useResourceB_F = competAbility_useResourceB_species0 .+ (traits_F .* (competAbility_useResourceB_species1 - competAbility_useResourceB_species0))
                ind_useResourceA_M = competAbility_useResourceA_species1 .+ ((1 .- traits_M) .* (competAbility_useResourceA_species0 - competAbility_useResourceA_species1))
                ind_useResourceB_M = competAbility_useResourceB_species0 .+ (traits_M .* (competAbility_useResourceB_species1 - competAbility_useResourceB_species0))
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

                # create empty data structures for keeping track of numbers offspring of parents (which is N in this function)
                # function offspring_numbers_object(N)
                #     return zeros(Int16, N)
                # end
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
                        match_strength = (exp(1) ^ ((-(traits_M[focal_male] - traits_F[mother])^2) / (2 * (pref_SD ^2))))
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
                    # determine fitness cost due to mate search (number of rejected males), which is zero in HZAM-sym paper 
                    search_fitness = (1-per_reject_cost) ^ rejects    # (in HZAM-sym paper, per_reject_cost = 0)
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
                
                # For later: add in here the option of tracking fitness?
                
                # determine survival fitnesses of daughters due to epistasis
                HI_daughters = calc_traits_additive(genotypes_daughters[:,1:trait_loci,:])
                epistasis_fitness_daughters = 1 .- (1 - w_hyb) .* (4 .* HI_daughters .* (1 .- HI_daughters)).^beta
                daughters_survive = epistasis_fitness_daughters .> rand(length(epistasis_fitness_daughters))
                # same for sons:
                HI_sons = calc_traits_additive(genotypes_sons[:,1:trait_loci,:])
                epistasis_fitness_sons = 1 .- (1 - w_hyb) .* (4 .* HI_sons .* (1 .- HI_sons)).^beta
                sons_survive = epistasis_fitness_sons .> rand(length(epistasis_fitness_sons))

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
            traits_all_inds = []
            species0_proportion = []
            species1_proportion = []
            HI_NL_all_inds = []
            species0_proportion_NL = []
            species1_proportion_NL = [] 
            if extinction  # whole simulation went extinct
                outcome = "extinction"
                @save string("simulation_data.",run_name,".jld2") outcome
            else  # no complete extinction
                # use trait loci to calculate HI of each individual
                traits_all_inds = [calc_traits_additive(genotypes_F[:,trait_loci_cols,:]); calc_traits_additive(genotypes_M[:,trait_loci_cols,:])]
                species0_proportion = sum(traits_all_inds .== 0) / length(traits_all_inds)
                species1_proportion = sum(traits_all_inds .== 1) / length(traits_all_inds)
                if species0_proportion >= 0.85 || species1_proportion >= 0.85
                    outcome = "one_species"
                elseif (species0_proportion + species1_proportion >= 0.85) && (species0_proportion >= 0.15) && (species1_proportion >= 0.15)
                    outcome = "two_species"
                else
                    outcome = "blended"
                end
                HI_neutral_all_inds = [calc_traits_additive(genotypes_F[:,neutral_loci_cols,:]); calc_traits_additive(genotypes_M[:,neutral_loci_cols,:])]
                species0_proportion_NL = sum(HI_neutral_all_inds .== 0) / length(HI_neutral_all_inds)
                species1_proportion_NL = sum(HI_neutral_all_inds .== 1) / length(HI_neutral_all_inds) 
                @save string("HZAM_Sym_results/simulation_data.",run_name,".jld2") outcome traits_all_inds species0_proportion species1_proportion HI_neutral_all_inds species0_proportion_NL species1_proportion_NL 
            end
            println(run_name, "  outcome was: ", outcome)
            outcome_array[i,j,k] = outcome
        end # of S_AM loop
    end # of w_hyb loop   
end # of replicate loop

outcome_array

for i in 1:size(outcome_array, 3)
    filename = string("HZAM_Sym_results/outcomeArray_set",set_name,"_ecolDiff",ecolDiff,"_growthrate",intrinsic_R,"_K",K_total,"_TL",trait_loci,"_gen",max_generations,"_rep",replications[i])
    CSV.write(filename, Tables.table(outcome_array[:,:,i]), writeheader=false)
end
