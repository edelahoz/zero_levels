#!/usr/bin/env python3

import sqlite3
import json


def create_tables(db_name: str):
    """Creates the required tables if they don't exist."""
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()

        # Table to store unique map combinations
        cursor.execute('''CREATE TABLE IF NOT EXISTS map_combinations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            maps TEXT NOT NULL,
                            description TEXT
                        )''')

        # Table to store iterations, linked to a map combination
        cursor.execute('''CREATE TABLE IF NOT EXISTS iterations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            combination_id INTEGER NOT NULL,
                            iteration INTEGER NOT NULL,
                            map_name TEXT NOT NULL,
                            monopole REAL NOT NULL,
                            dipole_ax REAL NOT NULL,
                            dipole_ay REAL NOT NULL,
                            dipole_az REAL NOT NULL,
                            FOREIGN KEY (combination_id) REFERENCES map_combinations(id) ON DELETE CASCADE
                        )''')

        conn.commit()

def convert_results_to_dict(maps, iterations):

    list_results = []
    for res_iter in iterations:
        dict_results = {}
        for idx, map_name in enumerate(maps):
            dict_results[map_name] = res_iter[idx * 4: (idx + 1) * 4] 
            if list_results:
                dict_results[map_name] += list_results[-1][map_name]
        list_results.append(dict_results)
    return list_results

def store_results(db_name: str, maps, iterations, description=""):
    """
    Stores results for a map combination.

    Parameters:
        maps (list of str): The list of map names in the combination.
        iterations (array): Array with the monopole and dipole values for each map in each iteration.
        description (str): Optional description for the map combination.
    """
    maps_json = json.dumps(sorted(maps))  # Store maps as sorted JSON for consistency
    list_iterations = convert_results_to_dict(maps, iterations)

    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()

        # Check if the combination already exists
        cursor.execute("SELECT id FROM map_combinations WHERE maps = ?", (maps_json,))
        result = cursor.fetchone()

        if result:
            combination_id = result[0]
        else:
            cursor.execute("INSERT INTO map_combinations (maps, description) VALUES (?, ?)", 
                           (maps_json, description))
            combination_id = cursor.lastrowid  # Get newly created ID
        
        cursor.execute("DELETE FROM iterations WHERE combination_id = ?", (combination_id,))
    
        # Store iterations
        for i, iteration in enumerate(list_iterations, start=1):
            for map_name, (monopole, dx, dy, dz) in iteration.items():
                cursor.execute("""
                    INSERT INTO iterations 
                    (combination_id, iteration, map_name, monopole, dipole_ax, dipole_ay, dipole_az) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (combination_id, i, map_name, monopole, dx, dy, dz))

        conn.commit()
        print(f"Stored {len(iterations)} iterations for map combination {maps} (ID: {combination_id})")

def read_results(db_name: str, maps, selected_map=None, last_iteration_only=False):
    """
    Retrieves stored iterations for a given map combination.

    Parameters:
        maps (list of str): The map combination to query.
        selected_map (str, optional): If provided, filters results to only this map.
        last_iteration_only (bool, optional): If True, retrieves only the last iteration.

    Returns:
        List of tuples containing (iteration, map_name, monopole, dipole_ax, dipole_ay, dipole_az)
    """
    maps_json = json.dumps(sorted(maps))  # Ensure consistent format

    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()

        # Find the combination ID
        cursor.execute("SELECT id FROM map_combinations WHERE maps = ?", (maps_json,))
        result = cursor.fetchone()

        if not result:
            print("No data found for this map combination.")
            return []
        
        combination_id = result[0]

        # If last iteration only, find the max iteration number
        iteration_filter = ""
        params = [combination_id]

        if last_iteration_only:
            cursor.execute("SELECT MAX(iteration) FROM iterations WHERE combination_id = ?", (combination_id,))
            last_iter = cursor.fetchone()[0]
            if last_iter is None:
                print("No iterations found.")
                return []
            iteration_filter = "AND iteration = ?"
            params.append(last_iter)

        # If filtering by a specific map
        map_filter = ""
        if selected_map:
            map_filter = "AND map_name = ?"
            params.append(selected_map)

        # Query results
        cursor.execute(f"""SELECT iteration, map_name, monopole, dipole_ax, dipole_ay, dipole_az 
                           FROM iterations 
                           WHERE combination_id = ? {iteration_filter} {map_filter}
                           ORDER BY iteration, map_name""", 
                       params)

        results = cursor.fetchall()
        
        if not results:
            print("No matching results found.")
            return []

        print(f"Results for {maps} (Map: {selected_map if selected_map else 'All'}) - Last Iteration: {last_iteration_only}")

        last_iteration = None
        for row in results:
            iteration, map_name, monopole, dx, dy, dz = row
            if iteration != last_iteration:
                print(f"\nIteration {iteration}:")
                last_iteration = iteration
            print(f"  {map_name}: Monopole={monopole}, Dipole=({dx}, {dy}, {dz})")
        
        return results

