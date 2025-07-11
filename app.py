from flask import Flask, render_template, request, jsonify
import math
import random
import os
from copy import deepcopy
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')


# Vehicle and Package classes
class Vehicle:
    def __init__(self, capacity, x=0, y=0):
        self.capacity = capacity
        self.added_packages = [] #later
        self.remaining_capacity = capacity #after assigning packages
        self.x = x
        self.y = y
    #function to assign a package to a vehicle only if it fits
    def assign_package(self, package):
        #if capacity left in vehicle fits the weight of package then add the package to added_packages
        if self.remaining_capacity >= package.weight:
            self.added_packages.append(package) #added if the condition is verified
            self.remaining_capacity -= package.weight #update the capacity of the vehicle
            return True
        return False

class Package:
    def __init__(self, x, y, weight, priority):
        self.destination = {'x': x, 'y': y}  # destination as a dict
        self.weight = weight
        self.priority = priority

# This Function is used to randomly assign packages to vehicles
def initial_random_state(vehicles_list, packages_list):
    for package in packages_list:
        vehicles = vehicles_list.copy()
        random.shuffle(vehicles)  # shuffle the vehicles to make sure its random
        added = False #flag to track if the package is assigned or not
        for vehicle in vehicles:
            if vehicle.assign_package(package): #add the package using the assign function
                added = True #set the flag
                break  # Stop when its assigned
        if not added:
            print(
                f" Warning: Package with weight {package.weight} could not be assigned to any vehicle and will be skipped.")

    # Debugging each vehicle after the assignment to see if packages were correctly assigned
    #for vehicle in vehicles_list:
     #   print(f"Vehicle with capacity {vehicle.capacity} now has packages:")
      #  for assigned_package in vehicle.added_packages:
       #     print(f"  Assigned Package: {assigned_package.destination} with weight {assigned_package.weight}")

    # at the end shuffle each vehicle's delivery route
    for vehicle in vehicles_list:
        random.shuffle(vehicle.added_packages)


@app.route('/')
def insertion_form():  # put application's code here
    return render_template('insertion.html')

# Route to upload file and process data
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and file.filename.endswith('.txt'):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure folder exists
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        with open(filepath, 'r') as f:
            content = f.read().strip()
        try:
            # Example format: "genetic 3{[capacity],[capacity],[capacity]} {[x,y,weight,priority],[x,y,weight,priority]}"
            algorithm, rest = content.split(' ', 1)
            vehicle_num_str, rest = rest.split('{', 1)
            vehicle_num = int(vehicle_num_str.strip())

            capacities_str, packages_str = rest.split('} {')
            capacities = [{"capacity": int(cap.strip('[]'))} for cap in capacities_str.strip('{}').split('],[')]

            # Parsing package data into a list of objects
            packages = []
            for pkg in packages_str.strip('{}').split('],['):
                parts = pkg.strip('[]').split(',')
                packages.append({
                    'x': float(parts[0]),
                    'y': float(parts[1]),
                    'weight': float(parts[2]),
                    'priority': int(parts[3])
                })

            # Process the data based on the selected algorithm
            vehicles_list = []
            for vehicle_data in capacities:
                vehicle = Vehicle(vehicle_data['capacity'])
                vehicles_list.append(vehicle)

            packages_list = []
            for package_data in packages:
                x = package_data['x']
                y = package_data['y']
                weight = package_data['weight']
                priority = package_data['priority']
                package = Package(x, y, weight, priority)
                packages_list.append(package)

            if algorithm == "genetic":

                best_fitness, best_individual, filtered_packages, vehicles_info ,skipped_packages,best_distance = geneticAlgorithm(vehicle_num, capacities, packages)

                # Format final readable solution
                vehicle_package_list = []
                for car in best_individual:
                    vehicle_data = []
                    for pkg_index in car:
                        pkg = filtered_packages[pkg_index]
                        vehicle_data.append({
                            'destination': f"({pkg['x']}, {pkg['y']})",
                            'weight': pkg['weight'],
                            'priority': pkg['priority']
                        })
                    vehicle_package_list.append(vehicle_data)

                # Return the full genetic result
                return jsonify({
                    "status": "ok",
                    "algorithm": "genetic",
                    "best_individual": best_individual,
                    "packages": filtered_packages,
                    "vehicles": vehicles_info,
                    "skipped_packages": skipped_packages,
                    "vehicle_package_list": vehicle_package_list,
                    "best_fitness": best_fitness,
                    "best_distance": best_distance
                })

            else:
                max_capacity = max(v.capacity for v in vehicles_list)
                skipped_packages = [p for p in packages_list if p.weight > max_capacity]
                packages_list = [p for p in packages_list if p.weight <= max_capacity]

                best_solution, best_distance = simulated_annealing(vehicles_list, packages_list)
                vehicle_package_list = convert_solution_to_list(best_solution)
                # Print the result
                print(f"The best solution found with total distance: {best_distance}")
                for i, vehicle in enumerate(best_solution):
                    print(f"Vehicle {i + 1}:")
                    for package in vehicle.added_packages:
                        print(f"  Package destination is: {package.destination} with weight: {package.weight}")

                print("\nConverted Vehicle-Package List (List Format):")
                for i, vehicle_packages in enumerate(vehicle_package_list):
                    print(f"Vehicle {i + 1}:")
                    for package in vehicle_packages:
                         print(f"  Package destination: {package['destination']} with weight: {package['weight']} and priority: {package['priority']}")

                # Return the best solution's total distance
                # Return the best solution's total distance and the vehicle-package list to the UI
                assigned_packages = []
                for vehicle in best_solution:
                    for package in vehicle.added_packages:
                        assigned_packages.append({
                            'destination': f"({package.destination['x']}, {package.destination['y']})",
                            'weight': package.weight,
                            'priority': package.priority
                        })

                # Prepare skipped packages for JSON
                skipped = [
                    {
                        'destination': f"({p.destination['x']}, {p.destination['y']})",
                        'weight': p.weight,
                        'priority': p.priority
                    } for p in skipped_packages
                ]

                return jsonify({
                    "status": "ok",
                    "algorithm": "simulated",
                    "best_distance": best_distance,
                    "vehicle_package_list": vehicle_package_list,
                    "vehicles": [v.capacity for v in vehicles_list],
                    "packages": assigned_packages,
                    "skipped_packages": skipped,

                })



        except Exception as e:
            return jsonify({"message": f"Failed to parse file: {str(e)}"}), 400
    else:
        return jsonify({"message": "Invalid file format"}), 400

@app.route('/results.html')
def show_results_page():
    return render_template('results.html')
@app.route('/userInsertion', methods=['POST'])
def userInsertion():
    data = request.get_json()
    vehicles_number = int(data['vehiclesNumber'])
    vehicles = data['vehicles']
    packages = data['packages']
    algorithm = data['algorithm']
    print(algorithm, vehicles_number, vehicles, packages)

    vehicles_list = []
    for vehicle_data in vehicles:
        vehicle = Vehicle(vehicle_data['capacity'])
        vehicles_list.append(vehicle)

    packages_list = []
    for package_data in packages:
        x = package_data['x']
        y = package_data['y']
        weight = package_data['weight']
        priority = package_data['priority']
        package = Package(x, y, weight, priority)
        packages_list.append(package)
    ##############################debug the initial random state###################################
    # Output the vehicles and their assigned packages
    # for i, vehicle in enumerate(vehicles_list):
    #     print(f"Vehicle {i + 1}:")
    #     print("  Assigned Packages:")
    #     for package in vehicle.assigned_packages:
    #         print(f"    Destination: {package.destination}, Weight: {package.weight}, Priority: {package.priority}")
        if algorithm == "genetic":

            best_fitness, best_individual, filtered_packages, vehicles_info, skipped_packages,best_distance = geneticAlgorithm(vehicles_number, vehicles, packages)

            # Format final readable solution
            vehicle_package_list = []
            for car in best_individual:
                vehicle_data = []
                for pkg_index in car:
                    pkg = filtered_packages[pkg_index]
                    vehicle_data.append({
                        'destination': f"({pkg['x']}, {pkg['y']})",
                        'weight': pkg['weight'],
                        'priority': pkg['priority']
                    })
                vehicle_package_list.append(vehicle_data)

            # Return the full genetic result
            return jsonify({
                "status": "ok",
                "algorithm": "genetic",
                "best_individual": best_individual,
                "packages": filtered_packages,
                "vehicles": vehicles_info,
                "skipped_packages": skipped_packages,
                "vehicle_package_list": vehicle_package_list,
                "best_fitness": best_fitness,
                "best_distance": best_distance
            })
        else:
            max_capacity = max(v.capacity for v in vehicles_list)
            skipped_packages = [p for p in packages_list if p.weight > max_capacity]
            packages_list = [p for p in packages_list if p.weight <= max_capacity]

            best_solution, best_distance = simulated_annealing(vehicles_list, packages_list)
            vehicle_package_list = convert_solution_to_list(best_solution)
            # Print the result
            print(f"The best solution found with total distance: {best_distance}")
            for i, vehicle in enumerate(best_solution):
                print(f"Vehicle {i + 1}:")
                for package in vehicle.added_packages:
                    print(f"  Package destination is: {package.destination} with weight: {package.weight}")

            print("\nConverted Vehicle-Package List (List Format):")
            for i, vehicle_packages in enumerate(vehicle_package_list):
                print(f"Vehicle {i + 1}:")
                for package in vehicle_packages:
                    print(
                        f"  Package destination: {package['destination']} with weight: {package['weight']} and priority: {package['priority']}")

            # Return the best solution's total distance
            # Return the best solution's total distance and the vehicle-package list to the UI
            assigned_packages = []
            for vehicle in best_solution:
                for package in vehicle.added_packages:
                    assigned_packages.append({
                        'destination': f"({package.destination['x']}, {package.destination['y']})",
                        'weight': package.weight,
                        'priority': package.priority
                    })

            # Prepare skipped packages for JSON
            skipped = [
                {
                    'destination': f"({p.destination['x']}, {p.destination['y']})",
                    'weight': p.weight,
                    'priority': p.priority
                } for p in skipped_packages
            ]

            return jsonify({
                "status": "ok",
                "algorithm": "simulated",
                "best_distance": best_distance,
                "vehicle_package_list": vehicle_package_list,
                "vehicles": [v.capacity for v in vehicles_list],
                "packages": assigned_packages,
                "skipped_packages": skipped
            })

    return jsonify({"status": "ok", "message": "File data processed!"})


def geneticAlgorithm(vehicles_number, vehicles, packages):
    # here is the algorithm for genetic
    # print(vehicles_number, vehicles, packages)
    #  Filter out oversized packages
    max_capacity = max(v['capacity'] for v in vehicles)

    oversized = [p for p in packages if p['weight'] > max_capacity]
    if oversized:
        for p in oversized:
            print(f"Skipping package with weight {p['weight']} (too large for any vehicle)")

    oversized = [
        {
            'x': p['x'],
            'y': p['y'],
            'weight': p['weight'],
            'priority': p['priority'],
            'destination': f"({p['x']}, {p['y']})"
        }
        for p in packages if p['weight'] > max_capacity
    ]
    packages[:] = [p for p in packages if p['weight'] <= max_capacity]

    if len(packages) == 0:
        return 0, [[] for _ in range(vehicles_number)], [], vehicles, oversized,0

    destinations = []
    # this will be needed to calculate the fitness function
    length = len(packages)
    for i in range(0, length):
        x1 = packages[i]['x']
        y1 = packages[i]['y']
        destinations.append((x1, y1))
    print("test1")
    # here we create the first generation

    # first we will create the individuals (the initial population)

    # Create the initial population by creating 70 valid individuals
    initial_population = create_initial_population(packages, vehicles_number, vehicles)
    print("the initial population is:")
    print(initial_population)
    # now we need to create a fitness function so we can find the parents
    # print(fitness_function(initial_population, packages))
    # print(choose_parents(initial_population, packages, fitness_function(initial_population, packages)))

    population = initial_population  # <- very important

    for i in range(500):  # inside the loop
        population = new_poplulation(population, vehicles_number, vehicles, packages)

    # (now outside the loop)
    # Evaluate final fitness and select best individual
    final_fitness = fitness_function(population, packages)
    fitness_population_pairs = list(zip(final_fitness, population))

    # Total number of packages
    total_assigned_packages = len(packages)

    # Try to find full-coverage individuals
    full_coverage_individuals = [
        pair for pair in fitness_population_pairs
        if sum(len(car) for car in pair[1]) == total_assigned_packages
    ]

    if full_coverage_individuals:
        full_coverage_individuals.sort(reverse=True, key=lambda x: x[0])
        best_fitness, best_individual = full_coverage_individuals[0]
    else:
        fitness_population_pairs.sort(reverse=True, key=lambda x: x[0])
        best_fitness, best_individual = fitness_population_pairs[0]

    # ====== NEW PART: Detect unassigned packages ======
    used_package_indexes = set(pkg for car in best_individual for pkg in car)
    all_package_indexes = set(range(len(packages)))
    missing_indexes = list(all_package_indexes - used_package_indexes)

    oversized = [packages[i] for i in missing_indexes]

    # Format for frontend
    oversized = [
        {
            'destination': f"({p['x']}, {p['y']})",
            'weight': p['weight'],
            'priority': p['priority']
        } for p in oversized
    ]
    best_distance = calculate_total_distance_for_individual(best_individual, packages)
    # ====== Final return ======
    return best_fitness, best_individual, packages, vehicles, oversized,best_distance
    # we need to make sure that the fitness takes priority in consideration
    # after the fitness completely done we start choosing parents
    #################################

    # print(destinations[i])
    # print(type(destinations[0]))


def calculate_total_distance_for_individual(individual, packages):
    total = 0
    depot = {'x': 0, 'y': 0}
    for car in individual:
        if not car:
            continue
        current_location = depot.copy()
        for pkg_index in car:
            next_location = {
                'x': packages[pkg_index]['x'],
                'y': packages[pkg_index]['y']
            }
            total += euclidean_distance_leen(current_location, next_location)
            current_location = next_location
        total += euclidean_distance_leen(current_location, depot)  # return to depot
    return total


def create_initial_population(packages, vehicles_number, vehicles):
    population_size = 70
    population = []
    attempts = 0
    max_attempts = 500

    while len(population) < population_size and attempts < max_attempts:
        individual = create_valid_individual(packages, vehicles_number, vehicles)
        attempts += 1
        if individual is not None:
            population.append(individual)

    if len(population) < population_size:
        print("only {len(population)} individuals created after {attempts} tries.")

    return population

def create_valid_individual(packages, vehicles_number, vehicles):
    packages = sorted(packages, key=lambda p: (-p['weight']))


    individual = [[] for _ in range(vehicles_number)]
    vehicle_weights = [0] * vehicles_number
    skipped = []  # <- Track skipped packages

    for i in range(len(packages)):
        placed = False
        car_index = list(range(vehicles_number))
        random.shuffle(car_index)

        for vehicle_index in car_index:
            if vehicle_weights[vehicle_index] + packages[i]['weight'] <= vehicles[vehicle_index]['capacity']:
                individual[vehicle_index].append(i)
                vehicle_weights[vehicle_index] += packages[i]['weight']
                placed = True
                break

        if not placed:
            skipped.append(i)

    if skipped:
        print(f"Skipped packages in this individual: {skipped}")

    return individual

# fitness function
def fitness_function(population, packages):
    fitness_array = []
    max_possible_length = calculate_max_possible_length(packages)

    for individual in population:
        total_priority_points = 0
        total_distance = 0

        for vehicle_route in individual:
            tour_distance = calculate_tour(vehicle_route, packages)
            total_distance += tour_distance

            # Priority per distance unit — better if priority is earned with less distance
            priority = calculate_priority_points(packages, vehicle_route)
            if tour_distance > 0:
                total_priority_points += priority / tour_distance
            else:
                total_priority_points += priority  # fallback in case of zero distance

        normalized_priority = total_priority_points / len(individual)
        distance_score = max(0.0, (max_possible_length - total_distance) / max_possible_length)

        fitness = 0.8 * distance_score + 0.2 * normalized_priority
        fitness_array.append(fitness)

    return fitness_array

def calculate_tour(tour, package): ################
    if not tour:
        return 0
    dist = 0
    current_location = {'x': 0, 'y': 0 }
    intial_location = {'x': 0, 'y': 0}
    for package_index in tour:
        next_location ={'x': package[package_index]['x'], 'y': package[package_index]['y'] } # the our will be based on the packages indexes sorting inside the cars
        dist += euclidean_distance_leen(current_location, next_location)
        current_location = next_location #from where we delivered the last package to the next location
    #add distance to go back to the origin
    dist += euclidean_distance_leen(current_location, intial_location)
    return dist


def calculate_priority_points(packages,tour):
    if len(tour) < 2:
        return 0.0  # No ordering to evaluate
    priority_points_for_the_tour = 0

    for i in range(len(tour) - 1):  # compare two packages at a time
        current_package_index = tour[i]
        next_package_index = tour[i + 1]

        current_priority = packages[current_package_index]['priority']
        next_priority = packages[next_package_index]['priority']

        if current_priority <= next_priority:
            # Good.high priority package is delivered before or at same time
            priority_points_for_the_tour += 1  # 1 point for good order

    percent_points = priority_points_for_the_tour / (len(tour) or 1) # if 1 2 3 4 5 then it is 1.0 good , it could be half etc...

    return percent_points


def calculate_max_possible_length(packages):
    if not packages:
        return 0

    depot = {'x': 0, 'y': 0}
    unvisited = packages[:]
    current = depot.copy()
    total_distance = 0

    while unvisited:
        # Find nearest unvisited package
        nearest = min(unvisited, key=lambda p: euclidean_distance_leen(current, {'x': p['x'], 'y': p['y']}))
        dist = euclidean_distance_leen(current, {'x': nearest['x'], 'y': nearest['y']})
        total_distance += dist
        current = {'x': nearest['x'], 'y': nearest['y']}
        unvisited.remove(nearest)

    # Return to depot
    total_distance += euclidean_distance_leen(current, depot)
    return total_distance



# choose parents function # the choice is based on the fitness array
def choose_parents(population, packages, fitness_array):
    percents_array = fitness_percent_array(fitness_array)
    #select parents (individuals) from the population and how likely they are choosen is based on fitness ... k=2 so two parents
    selected_parents = random.choices(population, weights=percents_array, k=2)
    attempts = 0
    while selected_parents[0] == selected_parents[1] and attempts < 10:
        selected_parents = random.choices(population, weights=percents_array, k=2)
        attempts += 1
        #the returned value is a list of two indivduals (same type as population but 2 individuals only
    return selected_parents




def fitness_percent_array(fitness_array):
    total_sum = sum(fitness_array)
    if total_sum > 0:
        return [fitness / total_sum for fitness in fitness_array]
    else:
        # fallback: assign uniform probability
        return [1 / len(fitness_array)] * len(fitness_array)





def new_poplulation(current_population, vehicles_number, vehicles, packages):
    new_population = []
    fitness_array = fitness_function(current_population, packages)

    i=0
    for i in range(34):
        parents =choose_parents(current_population, packages,fitness_array)
        print("test2")
        two_children = crossover_function(parents, vehicles_number, vehicles,packages)
        two_children[0] = mutate(two_children[0], vehicles, packages)
        two_children[1] = mutate(two_children[1], vehicles, packages)
        print("test3")
        new_population.append(two_children[0])
        new_population.append(two_children[1])
        #add the best two indivduals from last generation to the new one
        # pair the fitness with individuals so we can sort without losing the index
    fitness_population_pairs = list(zip(fitness_array, current_population))
    # sort fitness descending
    fitness_population_pairs.sort(reverse=True, key=lambda x: x[0])

    from_old_population1 = fitness_population_pairs[0][1]  # take the individual, not the fitness
    from_old_population2 = fitness_population_pairs[1][1]
    # add them to the new population so the population has 70 chromosomes
    new_population.append(from_old_population1)
    new_population.append(from_old_population2)

    return new_population

# crossover function
def crossover_function(parents, vehicles_number, vehicles, packages):
    import random

    split_point = random.randint(1, vehicles_number - 2) if vehicles_number > 2 else 0

    child1 = [car.copy() for car in parents[0][:split_point]]
    child2 = [car.copy() for car in parents[1][:split_point]]

    while len(child1) < vehicles_number:
        child1.append([])

    while len(child2) < vehicles_number:
        child2.append([])

    child1_flat = [pkg for car in child1 for pkg in car]
    child2_flat = [pkg for car in child2 for pkg in car]

    def try_assign_all(parent_from, child, child_flat):
        for car in parent_from:
            for pkg in car:
                if pkg in child_flat:
                    continue

                placed = False
                for _ in range(10):  # Try 10 times
                    vehicle_index = random.randint(0, vehicles_number - 1)
                    current_weight = sum(packages[p]['weight'] for p in child[vehicle_index])
                    if current_weight + packages[pkg]['weight'] <= vehicles[vehicle_index]['capacity']:
                        child[vehicle_index].append(pkg)
                        child_flat.append(pkg)
                        placed = True
                        break

                if not placed:
                    return False  # crossover failed

        return True  # crossover successful

    success1 = try_assign_all(parents[1], child1, child1_flat)
    success2 = try_assign_all(parents[0], child2, child2_flat)

    if not success1 or not success2:
        return [parents[0], parents[1]]  # fallback if any child failed

    return [child1, child2]


def mutate (child, vehicles, packages, mutation_rate=0.05): # high mutation rate so that if the capacity is not enough no swap and we are good
    if random.random() < mutation_rate:
        vehicles_num = len (child)
        if vehicles_num < 2:
            return child  # Can't swap between fewer than 2 cars
        #two random cars to swap the packages in them
        car_index = random.sample(range(vehicles_num),2)
        car1_idx = car_index[0]
        car2_idx = car_index[1]
        if child[car1_idx] and child[car2_idx]: #at least one pkg in both
            #random package
            pkg1_index = random.choice(range(len(child[car1_idx])))
            pkg2_index = random.choice(range(len(child[car2_idx])))
            pkg1 = child[car1_idx][pkg1_index]
            pkg2 = child[car2_idx][pkg2_index]

            #respects capacities?
            weight1 = packages[pkg1]['weight']
            weight2 = packages[pkg2]['weight']

            car1_current_weight = sum(packages[p]['weight'] for p in child[car1_idx])
            car2_current_weight = sum(packages[p]['weight'] for p in child[car2_idx])

            car1_new_weight = car1_current_weight - weight1 + weight2
            car2_new_weight = car2_current_weight - weight2 + weight1

            if car1_new_weight <= vehicles[car1_idx]['capacity'] and car2_new_weight <= vehicles[car2_idx]['capacity']:
                # Swap
                child[car1_idx][pkg1_index], child[car2_idx][pkg2_index] = child[car2_idx][pkg2_index], child[car1_idx][pkg1_index]
    return child


def euclidean_distance_leen(point1, point2):
    return math.sqrt((point2['x'] - point1['x']) ** 2 + (point2['y'] - point1['y']) ** 2)

############################simulated
# Euclidean distance function
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
#This function is used to change the routes of delivery
def change_route(vehicle):
    #if there are no packages assigned, then we can't do anything
    if not vehicle.added_packages:
        return
    optimized_packages = []
    #current location of vehicle
    current_location = (vehicle.x, vehicle.y)
    #copy of added packages so we can remove without changing the original
    packages = vehicle.added_packages.copy()
    #while there still packages
    while packages:
        #we find the closest package to reduce the distance, using the min between current location and the next package location
        closest_package = min(packages,key=lambda p: euclidean_distance(current_location, (p.destination['x'], p.destination['y'])))
        #add the closest package to the optimized packages list
        optimized_packages.append(closest_package)
        #update the current location to the next chosen package location
        current_location = (closest_package.destination['x'], closest_package.destination['y'])
        #then remove it from packages, since we're done with it
        packages.remove(closest_package)
    #update the list of packages added to the vehicle to the optimized one
    vehicle.added_packages = optimized_packages

# Generate a neighbor solution by randomly swapping packages between vehicles
def generate_new_solution(vehicles_list):
    new_solution = deepcopy(vehicles_list)
    #if there is only one vehicle then just change the route of packets.
    if len(new_solution) == 1:
        change_route(new_solution[0])
    else:
      #if there are more than one vehicle then we'll swap the packages
      #choose two random vehicles
      vehicle1 = random.choice(new_solution)
      vehicle2 = random.choice(new_solution)
    # make sure we didn't choose the same vehicle
      while vehicle1 == vehicle2:
        #if they're the same just rechoose a vehicle
        vehicle2 = random.choice(new_solution)
      #if there are packages assigned to both random chosen vehicles, then choose random packages
      if vehicle1.added_packages and vehicle2.added_packages:
        package1 = random.choice(vehicle1.added_packages)
        package2 = random.choice(vehicle2.added_packages)
        #if both packages can be fit in the two vehicles, then we can swap them
        fit1 = (vehicle1.remaining_capacity+package1.weight)-package2.weight
        fit2 = (vehicle2.remaining_capacity+package2.weight)-package1.weight
        #if it fits then remove p1 from v1 and p2 from v2 then add p1 to v2 and p2 to v1 (swap)
        if fit1 >= 0 and fit2 >= 0:
            vehicle1.added_packages.remove(package1)
            vehicle2.added_packages.remove(package2)
            vehicle1.added_packages.append(package2)
            vehicle2.added_packages.append(package1)

            # Update remaining capacities
            vehicle1.remaining_capacity = fit1
            vehicle2.remaining_capacity = fit2
        #print("Swapped packages!") ####debug
        #after swapping, perform the changing route to get a better new solution
        for vehicle in new_solution:
            change_route(vehicle)
    #return the final new solution
    return new_solution

# Calculate the total distance
#def Total_distance(vehicles_list):
#    total_distance = 0
#    for vehicle in vehicles_list:
#        #if there are packages assigned to each vehicle
#        if vehicle.added_packages:
#           #get the start location
#            current_location = (vehicle.x, vehicle.y)
#            #for each assigned package
#            for package in vehicle.added_packages:
#                #get their destination
#                package_location = (package.destination['x'], package.destination['y'])
#                #use Euclidean function to calc the distance between current location and the package destination
#                total_distance += euclidean_distance(current_location, package_location)
#                #update the current location to the package destination
#                current_location = package_location
            # the next comment is used when we consider that the vehicle return to the start after delivering the package
            #total_distance += euclidean_distance(current_location, (vehicle.x, vehicle.y))

# return total_distance
def Total_distance(vehicles_list):
    #initialize
    total_distance = 0
    penalty = 0

    for vehicle in vehicles_list:
        if vehicle.added_packages:
            #current location of vehicle
            current_location = (vehicle.x, vehicle.y)
            #calc the weight of the package = total capacity - remaining
            load = vehicle.capacity - vehicle.remaining_capacity
            #if the weight > capacity set penalty +1000
            if load > vehicle.capacity:
                penalty += 1000

            # Calculate total distance and check priority:
            previous_priority = None
            for package in vehicle.added_packages:
                package_location = (package.destination['x'], package.destination['y'])
                #calc the distance to the package
                distance = euclidean_distance(current_location, package_location)
                total_distance += distance
                #update the location to the package location
                current_location = package_location
                # If the current package has a higher priority then penalize for large distances
                if previous_priority is not None and package.priority < previous_priority:
                    if distance > 10:  # If the distance is > 10, add a small penalty
                        penalty += 50
                previous_priority = package.priority
            total_distance += euclidean_distance(current_location, (vehicle.x, vehicle.y))
    #add the penalty to the total distance
    return total_distance + penalty

#this function is used to calc the probability
def acceptance_probability(best, new, temperature):
    #e^((best-new)/temp)
    return math.exp((best - new) / temperature)

# Simulated Annealing function
def simulated_annealing(vehicles_list, packages_list):
    # call this function to get the initial state
    initial_random_state(vehicles_list, packages_list)
    #calc initial total distance
    initial_distance = Total_distance(vehicles_list)
    #copy.deepcopy(current_solution)
    #current sol is just the vehicles list
    current_solution = vehicles_list.copy()
    #for now the best sol is the same as current sol
    best_solution = current_solution.copy()
    #the same for best distance
    best_distance = initial_distance
    #just debug
    print(F"The initial distance is: {initial_distance}")

    temperature = 1000 #parameters defined by dr yazan
    cooling_rate = 0.95 # between 0.90 - 0.99
    iterations = 100
    #start
    while temperature > 1:
     for i in range(iterations):
        # Generate a new solution
        new_neighbor_solution = generate_new_solution(current_solution)
        #Calculate the new total distance for the new neighbor sol.
        new_neighbor_distance = Total_distance(new_neighbor_solution)
        if new_neighbor_distance < best_distance: #if the new distance is less than the current then of course change
            #better solution, so update the current and the best solution
            current_solution = new_neighbor_solution
            best_solution = current_solution
            best_distance = new_neighbor_distance
        else: #Calculate acceptance probability, to decide if we're accepting the new sol or not
            #use function to calc the probability
            accept = acceptance_probability(best_distance, new_neighbor_distance, temperature)
            #if the probability is > a random, then accept the new solution
            if accept > random.random():
                current_solution = new_neighbor_solution
        # Decrease temperature each iteration
        temperature = temperature * (1 - cooling_rate)

    return best_solution, best_distance

def convert_solution_to_list(solution):
    vehicle_package_list = []
    # Loop through the solution, which contains the vehicles and their assigned packages
    for vehicle in solution:
        # For each vehicle, we create a list of package details
        package_info = []
        for package in vehicle.added_packages:
            # Assuming package.destination is a tuple (x, y), format it as x, y
            x = package.destination['x']
            y = package.destination['y']
            package_info.append({
                'destination': f"({x}, {y})",  # Format destination as a string "(x, y)"
                'weight': package.weight,
                'priority': package.priority
            })
        # Append the list of packages for this vehicle
        vehicle_package_list.append(package_info)
    return vehicle_package_list



if __name__ == '__main__':
    app.run()
