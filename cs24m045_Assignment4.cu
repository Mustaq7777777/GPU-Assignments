// %%writefile main.cu
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

static const long long INF = (long long)4e18;

// Helper macro for error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
             << cudaGetErrorString(error) << endl; \
        exit(1); \
    } \
} while(0)

// Structure for an edge in the graph
struct Edge {
    int to, len;
};

// Structure for evacuation result
struct EvacuationResult {
    int path[10000];  
    long long drops[1000][3];  
    int path_size;
    int drops_size;
};

// Dijkstra's algorithm device function
__device__ void dijkstra(
    long long *shortestPath,
    bool *visitedNodes,  
    int *predecessorNodes,
    int *edgeTo,
    int startingNode,
    int totalCities,
    int *edgeLength,
    int *cityFirstEdge,
    int *cityNeighborCount
) {
    // Initialize all distances to infinity
    int i = 0;
    while (i < totalCities) {
        shortestPath[i] = INF;
        predecessorNodes[i] = -1;
        visitedNodes[i] = false;
        i++;
    }
    shortestPath[startingNode] = 0;
    
    // Main Dijkstra's loop
    for (int iteration = 0; iteration < totalCities; iteration++) {
        // Select node with minimum distance
        int chosenNode = -1;
        long long minimumDistance = INF;
        
        int node = 0;
        while (node < totalCities) {
            if (!visitedNodes[node] && shortestPath[node] < minimumDistance) {
                minimumDistance = shortestPath[node];
                chosenNode = node;
            }
            node++;
        }
        
        if (chosenNode == -1 || minimumDistance == INF) break;
        visitedNodes[chosenNode] = true;
        
        // Process all neighbors of chosen node
        int firstEdgeIndex = cityFirstEdge[chosenNode];
        int lastEdgeIndex = firstEdgeIndex + cityNeighborCount[chosenNode];
        
        for (int edgeIndex = firstEdgeIndex; edgeIndex < lastEdgeIndex; edgeIndex++) {
            int neighborNode = edgeTo[edgeIndex];
            int costToNeighbor = edgeLength[edgeIndex];
            if (shortestPath[chosenNode] + costToNeighbor < shortestPath[neighborNode]) {
                shortestPath[neighborNode] = shortestPath[chosenNode] + costToNeighbor;
                predecessorNodes[neighborNode] = chosenNode;
            }
        }
    }
}

// Random movement kernel with improved accuracy
__global__ void evacuateKernelRandom(
    EvacuationResult *d_results,
    bool *d_visited,
    unsigned int seed,
    int totalCities, int totalShelters, int totalPopulated, int elderlyMaxDistance,
    int *d_populationCities, int *d_populationPrime, int *d_populationElderly,
    int *d_shelterLocations, int *d_shelterCapacity,
    int *d_edgeDestinations, int *d_edgeLengths, int *d_edgeOffsets, int *d_edgeCount
) {
    int populationIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (populationIndex >= totalPopulated) return;

    // Initialize custom RNG
    unsigned int randomState = seed + populationIndex;

    // Get thread-specific workspace
    bool *isVisited = &d_visited[populationIndex * totalCities];

    // Setup initial state
    int sourceLocation = d_populationCities[populationIndex];
    long long primeRemaining = d_populationPrime[populationIndex];
    long long elderlyRemaining = d_populationElderly[populationIndex];
    int currentLocation = sourceLocation;

    EvacuationResult &result = d_results[populationIndex];
    result.path_size = 0;
    result.drops_size = 0;
    result.path[result.path_size++] = currentLocation;

    // Initialize visited array
    int initIndex = 0;
    while (initIndex < totalCities) {
        isVisited[initIndex] = false;
        initIndex++;
    }
    isVisited[currentLocation] = true;

    // Track elderly movement distance
    int elderlyTravelledDistance = 0;
    int firstNonShelterFind = -1;

    while ((primeRemaining > 0 || elderlyRemaining > 0) && result.path_size < 10000) {
        // Find if current city has a shelter
        int shelterAtCurrent = -1;
        for (int j = 0; j < totalShelters; j++) {
            if (d_shelterLocations[j] == currentLocation) {
                shelterAtCurrent = j;
                break;
            }
        }
        
        if (shelterAtCurrent >= 0) {
            int currentCapacity = atomicAdd(&d_shelterCapacity[shelterAtCurrent], 0);
            
            // Handle elderly first
            if (elderlyRemaining > 0 && currentCapacity > 0) {
                int previousCapacity, updatedCapacity, elderlyToPlace;
                do {
                    previousCapacity = atomicAdd(&d_shelterCapacity[shelterAtCurrent], 0);
                    elderlyToPlace = min(elderlyRemaining, (long long)previousCapacity);
                    updatedCapacity = previousCapacity - elderlyToPlace;
                } while (elderlyToPlace > 0 && atomicCAS(&d_shelterCapacity[shelterAtCurrent], previousCapacity, updatedCapacity) != previousCapacity);
                
                elderlyRemaining -= elderlyToPlace;
                if (elderlyToPlace > 0 && result.drops_size < 1000) {
                    result.drops[result.drops_size][0] = currentLocation;
                    result.drops[result.drops_size][1] = 0;
                    result.drops[result.drops_size][2] = elderlyToPlace;
                    result.drops_size++;
                }
            }
            
            // Handle prime-age people
            if (primeRemaining > 0 && currentCapacity > 0) {
                int previousCapacity, updatedCapacity, primeToPlace;
                do {
                    previousCapacity = atomicAdd(&d_shelterCapacity[shelterAtCurrent], 0);
                    primeToPlace = min(primeRemaining, (long long)previousCapacity);
                    updatedCapacity = previousCapacity - primeToPlace;
                } while (primeToPlace > 0 && atomicCAS(&d_shelterCapacity[shelterAtCurrent], previousCapacity, updatedCapacity) != previousCapacity);
                
                primeRemaining -= primeToPlace;
                if (primeToPlace > 0 && result.drops_size < 1000) {
                    result.drops[result.drops_size][0] = currentLocation;
                    result.drops[result.drops_size][1] = primeToPlace;
                    result.drops[result.drops_size][2] = 0;
                    result.drops_size++;
                }
            }
        } else if (firstNonShelterFind < 0) {
            firstNonShelterFind = currentLocation;
        }

        if (primeRemaining == 0 && elderlyRemaining == 0) break;

        // Collect adjacent unvisited cities
        int edgeStart = d_edgeOffsets[currentLocation];
        int edgeFinish = edgeStart + d_edgeCount[currentLocation];
        int adjacentNodes[64], adjacentDistances[64], adjacentCount = 0;
        
        for (int e = edgeStart; e < edgeFinish && adjacentCount < 64; e++) {
            int nextNode = d_edgeDestinations[e];
            if (!isVisited[nextNode]) {
                adjacentNodes[adjacentCount] = nextNode;
                adjacentDistances[adjacentCount] = d_edgeLengths[e];
                adjacentCount++;
            }
        }
        if (adjacentCount == 0) break;

        // Randomly select next destination
        int selectedIndex = (randomState * 1103515245 + 12345) % adjacentCount;
        randomState = selectedIndex;
        int nextDestination = adjacentNodes[selectedIndex];
        int distanceToNext = adjacentDistances[selectedIndex];

        // Handle elderly distance constraint
        if (elderlyRemaining > 0 && elderlyTravelledDistance + distanceToNext > elderlyMaxDistance) {
            // Drop elderly at current location
            if (result.drops_size < 1000) {
                result.drops[result.drops_size][0] = currentLocation;
                result.drops[result.drops_size][1] = 0;
                result.drops[result.drops_size][2] = elderlyRemaining;
                result.drops_size++;
            }
            elderlyRemaining = 0;
        }

        // Continue to next location
        elderlyTravelledDistance += (elderlyRemaining > 0 ? distanceToNext : 0);
        currentLocation = nextDestination;
        isVisited[currentLocation] = true;
        if (result.path_size < 10000) {
            result.path[result.path_size++] = currentLocation;
        }
    }

    // Handle remaining prime-age people
    if (primeRemaining > 0 && result.drops_size < 1000) {
        int dropLocation = (firstNonShelterFind >= 0 ? firstNonShelterFind : currentLocation);
        result.drops[result.drops_size][0] = dropLocation;
        result.drops[result.drops_size][1] = primeRemaining;
        result.drops[result.drops_size][2] = 0;
        result.drops_size++;
        primeRemaining = 0;
    }
}

// Modified main kernel with proper parameter ordering
__global__ void evacuateKernel(
    int *d_populationCities, int *d_populationPrime, int *d_populationElderly,
    int *d_shelterLocations, int *d_shelterCapacity,
    int *d_cityMapping, int cityCount, int shelterCount, int populationCount, int elderlyMaxDistance,
    int *d_edgeDestinations, int *d_edgeLengths, int *d_edgeOffsets, int *d_edgeCount,
    EvacuationResult *d_results,
    long long *d_distances,
    int *d_previous,
    bool *d_visitedNodes,
    int *d_pathSegments,
    bool *d_shelterUsed,
    int *d_firstAppearance,
    bool *d_hasDrops,
    int *d_dropOrderIndex,
    long long *d_dropStorage
) {
    int populationIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (populationIndex >= populationCount) return;
    
    // Get pointers to this thread's memory space
    long long *dropStorage = &d_dropStorage[populationIndex * cityCount * 3];
    long long *distances = &d_distances[populationIndex * cityCount];
    int *previous = &d_previous[populationIndex * cityCount];
    bool *visitedNodes = &d_visitedNodes[populationIndex * cityCount];
    int *pathSegments = &d_pathSegments[populationIndex * cityCount];
    bool *shelterUsed = &d_shelterUsed[populationIndex * shelterCount];
    int *firstAppearance = &d_firstAppearance[populationIndex * cityCount];
    bool *hasDrops = &d_hasDrops[populationIndex * cityCount];
    int *dropOrderIndex = &d_dropOrderIndex[populationIndex * cityCount];


    // Set up shelter tracking
    int usedIndex = 0;
    while (usedIndex < shelterCount) {
        shelterUsed[usedIndex] = false;
        usedIndex++;
    }
    int totalUsed = 0;
    
    // Find first non-shelter city on path
    int firstNonShelterFind = -1;
    
    // Set up appearance tracking
    for (int idx = 0; idx < cityCount; idx++) {
        firstAppearance[idx] = -1;
    }
    
    // Set up drop tracking
    int dropIdx = 0;
    while (dropIdx < cityCount) {
        hasDrops[dropIdx] = false;
        dropIdx++;
    }

    // Initialize city tracking variables
    int startCity = d_populationCities[populationIndex];
    long long primeRemaining = d_populationPrime[populationIndex];
    long long elderlyRemaining = d_populationElderly[populationIndex];
    int activeCity = startCity;
    
    // Initialize result storage
    EvacuationResult &output = d_results[populationIndex];
    output.path_size = 0;
    output.drops_size = 0;
    
    // Add starting city to path
    output.path[output.path_size++] = activeCity;
    
    while ((primeRemaining > 0 || elderlyRemaining > 0) && totalUsed < shelterCount && output.path_size < 10000) {
        // Perform Dijkstra's algorithm
        dijkstra(distances, visitedNodes, previous, d_edgeDestinations, 
                activeCity, cityCount, d_edgeLengths, d_edgeOffsets, d_edgeCount);
        
        // Locate the closest available shelter
        long long bestRoute = INF;
        int targetCity = -1, targetIndex = -1;
        
        for (int shelterIdx = 0; shelterIdx < shelterCount; shelterIdx++) {
            if (shelterUsed[shelterIdx]) continue;
            
            int shelterLocation = d_shelterLocations[shelterIdx];
            int remainingCapacity = 0;
            
            // Check shelter capacity
            remainingCapacity = atomicAdd(&d_shelterCapacity[shelterIdx], 0);
            
            if (remainingCapacity > 0 && distances[shelterLocation] < bestRoute) {
                bestRoute = distances[shelterLocation];
                targetCity = shelterLocation;
                targetIndex = shelterIdx;
            }
        }
        
        if (targetIndex < 0) break; // No accessible shelter
        
        shelterUsed[targetIndex] = true;
        totalUsed++;
        
        // Reconstruct the route segment
        int segmentIdx = 0;
        
        for (int vertex = targetCity; vertex != -1; vertex = previous[vertex]) {
            pathSegments[segmentIdx++] = vertex;
            if (vertex == activeCity) break;
        }
        
        if (pathSegments[segmentIdx - 1] != activeCity) break; // Invalid segment
        
        // Find non-shelter cities in segment
        for (int i = segmentIdx - 2; i >= 0; i--) {
            int currentIdx = pathSegments[i];
            bool hasShelter = false;
            
            for (int shelterIdx = 0; shelterIdx < shelterCount; shelterIdx++) {
                if (d_shelterLocations[shelterIdx] == currentIdx) {
                    hasShelter = true;
                    break;
                }
            }
            
            if (!hasShelter && firstNonShelterFind == -1) {
                firstNonShelterFind = currentIdx;
            }
        }
        
        // Invert the segment for proper order
        int reverseIdx = 0;
        while (reverseIdx < segmentIdx / 2) {
            int temporary = pathSegments[reverseIdx];
            pathSegments[reverseIdx] = pathSegments[segmentIdx - 1 - reverseIdx];
            pathSegments[segmentIdx - 1 - reverseIdx] = temporary;
            reverseIdx++;
        }
        
        // Add segment to overall path (excluding the first)
        for (int step = 1; step < segmentIdx; step++) {
            if (output.path_size < 10000) {
                output.path[output.path_size++] = pathSegments[step];
            } else {
                // Path size maximum reached
                break;
            }
        }
        
        // Calculate elderly travel limit
        long long distanceSum = 0;
        int farthestPossible = pathSegments[0];
        
        // Track furthest non-shelter city for elderly
        int farthestNonShelterOption = -1;
        
        for (int step = 0; step + 1 < segmentIdx; step++) {
            int fromCity = pathSegments[step], toCity = pathSegments[step + 1];
            int edgeCost = -1;
            
            // Look up edge cost
            int initialEdge = d_edgeOffsets[fromCity];
            int finalEdge = initialEdge + d_edgeCount[fromCity];
            
            for (int edge = initialEdge; edge < finalEdge; edge++) {
                if (d_edgeDestinations[edge] == toCity) {
                    edgeCost = d_edgeLengths[edge];
                    break;
                }
            }
            
            distanceSum += edgeCost;
            if (distanceSum <= elderlyMaxDistance) {
                farthestPossible = toCity;
                
                // Check if this is a non-shelter location
                bool isShelterHere = false;
                for (int shelterIdx = 0; shelterIdx < shelterCount; shelterIdx++) {
                    if (d_shelterLocations[shelterIdx] == toCity) {
                        isShelterHere = true;
                        break;
                    }
                }
                
                if (!isShelterHere) {
                    farthestNonShelterOption = toCity;
                }
            } else {
                break;
            }
        }
        
        bool elderlyReachable = (farthestPossible == targetCity);
        
        // Place elderly who can't reach the shelter
        if (!elderlyReachable && elderlyRemaining > 0) {
            // Use the furthest non-shelter city if available
            if (farthestNonShelterOption != -1) {
                if (output.drops_size < 1000) {
                    output.drops[output.drops_size][0] = farthestNonShelterOption;
                    output.drops[output.drops_size][1] = 0;
                    output.drops[output.drops_size][2] = elderlyRemaining;
                    output.drops_size++;
                }
            } else {
                // Place at furthest reachable city
                if (output.drops_size < 1000) {
                    output.drops[output.drops_size][0] = farthestPossible;
                    output.drops[output.drops_size][1] = 0;
                    output.drops[output.drops_size][2] = elderlyRemaining;
                    output.drops_size++;
                }
            }
            elderlyRemaining = 0;
        } else if (elderlyReachable && elderlyRemaining > 0) {
            // Attempt to place elderly in shelter
            int currentShelterCap, newShelterCap, elderlyPlaced;
            do {
                currentShelterCap = atomicAdd(&d_shelterCapacity[targetIndex], 0);
                elderlyPlaced = min(elderlyRemaining, (long long)currentShelterCap);
                newShelterCap = currentShelterCap - elderlyPlaced;
            } while (elderlyPlaced > 0 && atomicCAS(&d_shelterCapacity[targetIndex], currentShelterCap, newShelterCap) != currentShelterCap);
            
            elderlyRemaining -= elderlyPlaced;
            
            if (elderlyPlaced > 0 && output.drops_size < 1000) {
                output.drops[output.drops_size][0] = targetCity;
                output.drops[output.drops_size][1] = 0;
                output.drops[output.drops_size][2] = elderlyPlaced;
                output.drops_size++;
            }
            
            if (elderlyRemaining > 0) {
                // Place remaining elderly
                if (farthestNonShelterOption != -1) {
                    if (output.drops_size < 1000) {
                        output.drops[output.drops_size][0] = farthestNonShelterOption;
                        output.drops[output.drops_size][1] = 0;
                        output.drops[output.drops_size][2] = elderlyRemaining;
                        output.drops_size++;
                    }
                } else {
                    // Check the overall path for non-shelter options
                    bool dropLocationFound = false;
                    int pathIdx = 0;
                    while (pathIdx < output.path_size) {
                        int cityOnPath = output.path[pathIdx];
                        bool isShelterHere = false;
                        for (int shelterIdx = 0; shelterIdx < shelterCount; shelterIdx++) {
                            if (d_shelterLocations[shelterIdx] == cityOnPath) {
                                isShelterHere = true;
                                break;
                            }
                        }
                        
                        if (!isShelterHere && output.drops_size < 1000) {
                            output.drops[output.drops_size][0] = cityOnPath;
                            output.drops[output.drops_size][1] = 0;
                            output.drops[output.drops_size][2] = elderlyRemaining;
                            output.drops_size++;
                            dropLocationFound = true;
                            break;
                        }
                        pathIdx++;
                    }
                    
                    // Default drop location
                    if (!dropLocationFound && output.drops_size < 1000) {
                        output.drops[output.drops_size][0] = targetCity;
                        output.drops[output.drops_size][1] = 0;
                        output.drops[output.drops_size][2] = elderlyRemaining;
                        output.drops_size++;
                    }
                }
                elderlyRemaining = 0;
            }
        }
        
        // Place prime-age people
        if (primeRemaining > 0) {
            // Update shelter capacity atomically
            int currentShelterCap, newShelterCap, primePlaced;
            do {
                currentShelterCap = atomicAdd(&d_shelterCapacity[targetIndex], 0);
                primePlaced = min(primeRemaining, (long long)currentShelterCap);
                newShelterCap = currentShelterCap - primePlaced;
            } while (primePlaced > 0 && atomicCAS(&d_shelterCapacity[targetIndex], currentShelterCap, newShelterCap) != currentShelterCap);
            
            primeRemaining -= primePlaced;
            
            if (primePlaced > 0 && output.drops_size < 1000) {
                output.drops[output.drops_size][0] = targetCity;
                output.drops[output.drops_size][1] = primePlaced;
                output.drops[output.drops_size][2] = 0;
                output.drops_size++;
            }
        }
        
        activeCity = targetCity;
    }
    
    // Handle leftover elderly at the end
    if (elderlyRemaining > 0) {
        long long distanceSum = 0;
        int farthestPossible = output.path[0];
        int farthestNonShelterOption = -1;
        
        for (int step = 0; step + 1 < output.path_size; step++) {
            int fromCity = output.path[step], toCity = output.path[step + 1];
            int edgeCost = -1;
            
            // Look up edge cost
            int initialEdge = d_edgeOffsets[fromCity];
            int finalEdge = initialEdge + d_edgeCount[fromCity];
            
            for (int edge = initialEdge; edge < finalEdge; edge++) {
                if (d_edgeDestinations[edge] == toCity) {
                    edgeCost = d_edgeLengths[edge];
                    break;
                }
            }
            
            distanceSum += edgeCost;
            if (distanceSum <= elderlyMaxDistance) {
                farthestPossible = toCity;
                
                // Check if this is a non-shelter location
                bool isShelterHere = false;
                for (int shelterIdx = 0; shelterIdx < shelterCount; shelterIdx++) {
                    if (d_shelterLocations[shelterIdx] == toCity) {
                        isShelterHere = true;
                        break;
                    }
                }
                
                if (!isShelterHere) {
                    farthestNonShelterOption = toCity;
                }
            } else {
                break;
            }
        }
        
        // Place at furthest non-shelter if available
        if (farthestNonShelterOption != -1 && output.drops_size < 1000) {
            output.drops[output.drops_size][0] = farthestNonShelterOption;
            output.drops[output.drops_size][1] = 0;
            output.drops[output.drops_size][2] = elderlyRemaining;
            output.drops_size++;
        } else if (output.drops_size < 1000) {
            // Place at furthest reachable
            output.drops[output.drops_size][0] = farthestPossible;
            output.drops[output.drops_size][1] = 0;
            output.drops[output.drops_size][2] = elderlyRemaining;
            output.drops_size++;
        }
        elderlyRemaining = 0;
    }
    
    // Handle leftover prime-age people
    if (primeRemaining > 0) {
        // Use first non-shelter if found
        if (firstNonShelterFind != -1 && output.drops_size < 1000) {
            output.drops[output.drops_size][0] = firstNonShelterFind;
            output.drops[output.drops_size][1] = primeRemaining;
            output.drops[output.drops_size][2] = 0;
            output.drops_size++;
        } else {
            // Find any non-shelter in the path
            bool locationFound = false;
            int pathIdx = 0;
            while (pathIdx < output.path_size) {
                int cityOnPath = output.path[pathIdx];
                bool isShelterHere = false;
                for (int shelterIdx = 0; shelterIdx < shelterCount; shelterIdx++) {
                    if (d_shelterLocations[shelterIdx] == cityOnPath) {
                        isShelterHere = true;
                        break;
                    }
                }
                
                if (!isShelterHere && output.drops_size < 1000) {
                    output.drops[output.drops_size][0] = cityOnPath;
                    output.drops[output.drops_size][1] = primeRemaining;
                    output.drops[output.drops_size][2] = 0;
                    output.drops_size++;
                    locationFound = true;
                    break;
                }
                pathIdx++;
            }
            
            // Last resort: current city
            if (!locationFound && output.drops_size < 1000) {
                output.drops[output.drops_size][0] = activeCity;
                output.drops[output.drops_size][1] = primeRemaining;
                output.drops[output.drops_size][2] = 0;
                output.drops_size++;
            }
        }
        primeRemaining = 0;
    }
    
    // Track each city's first appearance
    for (int i = 0; i < output.path_size; i++) {
        int cityOnPath = output.path[i];
        if (firstAppearance[cityOnPath] == -1) {
            firstAppearance[cityOnPath] = i;
        }
    }
    
    // Merge drops at same city
    for (int i = 0; i < output.drops_size; i++) {
        int dropCity = output.drops[i][0];
        long long primeAgeCount = output.drops[i][1];
        long long elderlyCount = output.drops[i][2];
        
        if (!hasDrops[dropCity]) {
            dropStorage[dropCity * 3] = dropCity;
            dropStorage[dropCity * 3 + 1] = primeAgeCount;
            dropStorage[dropCity * 3 + 2] = elderlyCount;
            hasDrops[dropCity] = true;
        } else {
            dropStorage[dropCity * 3 + 1] += primeAgeCount;
            dropStorage[dropCity * 3 + 2] += elderlyCount;
        }
    }
    
    // Prepare ordered drops
    int dropCounter = 0;
    
    for (int i = 0; i < cityCount; i++) {
        if (hasDrops[i] && firstAppearance[i] != -1) {
            dropOrderIndex[dropCounter * 4] = firstAppearance[i];
            dropOrderIndex[dropCounter * 4 + 1] = (int)dropStorage[i * 3];
            dropOrderIndex[dropCounter * 4 + 2] = (int)dropStorage[i * 3 + 1];
            dropOrderIndex[dropCounter * 4 + 3] = (int)dropStorage[i * 3 + 2];
            dropCounter++;
        }
    }
    
    // Sort drops by path position
    for (int i = 0; i < dropCounter - 1; i++) {
        for (int j = 0; j < dropCounter - i - 1; j++) {
            if (dropOrderIndex[j * 4] > dropOrderIndex[(j + 1) * 4]) {
                // Exchange elements
                for (int k = 0; k < 4; k++) {
                    int temporary = dropOrderIndex[j * 4 + k];
                    dropOrderIndex[j * 4 + k] = dropOrderIndex[(j + 1) * 4 + k];
                    dropOrderIndex[(j + 1) * 4 + k] = temporary;
                }
            }
        }
    }
    
    // Generate final drops array
    output.drops_size = 0;
    for (int i = 0; i < dropCounter && i < 1000; i++) {
        output.drops[output.drops_size][0] = dropOrderIndex[i * 4 + 1];
        output.drops[output.drops_size][1] = dropOrderIndex[i * 4 + 2];
        output.drops[output.drops_size][2] = dropOrderIndex[i * 4 + 3];
        output.drops_size++;
    }
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <in> <out>\n";
        return 1;
    }
    
    ifstream in(argv[1]);
    if (!in) {
        cerr << "Cannot open " << argv[1] << "\n";
        return 1;
    }

    int N, R;
    in >> N >> R;
    
    // Read graph
    vector<vector<Edge>> adj(N);
    for (int i = 0; i < R; i++) {
        int u, v, l, c;
        in >> u >> v >> l >> c;
        adj[u].push_back({v, l});
        adj[v].push_back({u, l});
    }

    // Read shelters
    int S;
    in >> S;
    vector<int> shelterCity(S), shelterCap(S), origCap(S);
    unordered_map<int, int> city2sidx;
    
    for (int i = 0; i < S; i++) {
        in >> shelterCity[i] >> shelterCap[i];
        origCap[i] = shelterCap[i];
        city2sidx[shelterCity[i]] = i;
    }

    // Read populated cities
    int P;
    in >> P;
    vector<int> popCity(P), popP(P), popE(P);
    
    for (int i = 0; i < P; i++) {
        in >> popCity[i] >> popP[i] >> popE[i];
    }
    
    int maxDistElder;
    in >> maxDistElder;
    in.close();

    // Prepare CUDA memory for graph
    vector<int> edges_to, edges_len, edge_offsets, edge_count;
    edge_offsets.resize(N);
    edge_count.resize(N);
    
    int edgeIndex = 0;
    while (edgeIndex < N) {
        edge_offsets[edgeIndex] = edges_to.size();
        edge_count[edgeIndex] = adj[edgeIndex].size();
        
        int innerIndex = 0;
        while (innerIndex < adj[edgeIndex].size()) {
            edges_to.push_back(adj[edgeIndex][innerIndex].to);
            edges_len.push_back(adj[edgeIndex][innerIndex].len);
            innerIndex++;
        }
        edgeIndex++;
    }
    
    vector<int> city2sidx_vec(N, -1);
    for (const auto& [city, idx] : city2sidx) {
        city2sidx_vec[city] = idx;
    }
    
    // Determine which algorithm to use
    bool useRandomMovement = (N > 1000);

    vector<EvacuationResult> results(P);
    vector<int> updatedShelterCap(S);

    
    if (!useRandomMovement) {
        // Original approach for small datasets (N <= 1000)
        
        // Create device memory
        int *d_populationCities, *d_populationPrime, *d_populationElderly;
        int *d_shelterLocations, *d_shelterCapacity;
        int *d_cityMapping;
        int *d_edgeDestinations, *d_edgeLengths, *d_edgeOffsets, *d_edgeCount;
        EvacuationResult *d_results;
        
        // Workspace arrays
        long long *d_distances;
        int *d_previous;
        bool *d_visitedNodes;
        int *d_pathSegments;
        bool *d_shelterUsed;
        int *d_firstAppearance;
        bool *d_hasDrops;
        int *d_dropOrderIndex;
        long long *d_dropStorage;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_populationCities, P * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_populationPrime, P * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_populationElderly, P * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_shelterLocations, S * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_shelterCapacity, S * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cityMapping, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeDestinations, edges_to.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeLengths, edges_len.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeOffsets, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeCount, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_results, P * sizeof(EvacuationResult)));
        
        // Allocate workspace
        CUDA_CHECK(cudaMalloc(&d_distances, P * N * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_previous, P * N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_visitedNodes, P * N * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_pathSegments, P * N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_shelterUsed, P * S * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_firstAppearance, P * N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_hasDrops, P * N * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_dropOrderIndex, P * N * 4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dropStorage, P * N * 3 * sizeof(long long)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_populationCities, popCity.data(), P * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_populationPrime, popP.data(), P * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_populationElderly, popE.data(), P * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_shelterLocations, shelterCity.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_shelterCapacity, shelterCap.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cityMapping, city2sidx_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeDestinations, edges_to.data(), edges_to.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeLengths, edges_len.data(), edges_len.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeOffsets, edge_offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeCount, edge_count.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (P + threadsPerBlock - 1) / threadsPerBlock;
        
        evacuateKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_populationCities, d_populationPrime, d_populationElderly,
            d_shelterLocations, d_shelterCapacity,
            d_cityMapping, N, S, P, maxDistElder,
            d_edgeDestinations, d_edgeLengths, d_edgeOffsets, d_edgeCount,
            d_results,
            d_distances, d_previous, d_visitedNodes, d_pathSegments, d_shelterUsed,
            d_firstAppearance, d_hasDrops, d_dropOrderIndex, d_dropStorage
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Clean up workspace arrays
        CUDA_CHECK(cudaFree(d_distances));
        CUDA_CHECK(cudaFree(d_previous));
        CUDA_CHECK(cudaFree(d_visitedNodes));
        CUDA_CHECK(cudaFree(d_pathSegments));
        CUDA_CHECK(cudaFree(d_shelterUsed));
        CUDA_CHECK(cudaFree(d_firstAppearance));
        CUDA_CHECK(cudaFree(d_hasDrops));
        CUDA_CHECK(cudaFree(d_dropOrderIndex));
        CUDA_CHECK(cudaFree(d_dropStorage));
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(results.data(), d_results, P * sizeof(EvacuationResult), 
                              cudaMemcpyDeviceToHost));
        
        // Copy back final shelter capacities
        CUDA_CHECK(cudaMemcpy(updatedShelterCap.data(), d_shelterCapacity, S * sizeof(int), 
                              cudaMemcpyDeviceToHost));
        
        // Clean up
        CUDA_CHECK(cudaFree(d_populationCities));
        CUDA_CHECK(cudaFree(d_populationPrime));
        CUDA_CHECK(cudaFree(d_populationElderly));
        CUDA_CHECK(cudaFree(d_shelterLocations));
        CUDA_CHECK(cudaFree(d_shelterCapacity));
        CUDA_CHECK(cudaFree(d_cityMapping));
        CUDA_CHECK(cudaFree(d_edgeDestinations));
        CUDA_CHECK(cudaFree(d_edgeLengths));
        CUDA_CHECK(cudaFree(d_edgeOffsets));
        CUDA_CHECK(cudaFree(d_edgeCount));
        CUDA_CHECK(cudaFree(d_results));
        
    } else {
        // Random movement for large datasets (N > 1000)
        
        // Create device memory
        int *d_populationCities, *d_populationPrime, *d_populationElderly;
        int *d_shelterLocations, *d_shelterCapacity;
        int *d_edgeDestinations, *d_edgeLengths, *d_edgeOffsets, *d_edgeCount;
        EvacuationResult *d_results;
        
        // Only need visited array for random movement
        bool *d_visited;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_shelterCapacity, S * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeDestinations, edges_to.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeLengths, edges_len.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeOffsets, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edgeCount, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_results, P * sizeof(EvacuationResult)));
        CUDA_CHECK(cudaMalloc(&d_populationCities, P * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_populationPrime, P * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_populationElderly, P * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_shelterLocations, S * sizeof(int)));

        
        // Minimal workspace for random movement
        CUDA_CHECK(cudaMalloc(&d_visited, P * N * sizeof(bool)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_shelterCapacity, shelterCap.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeDestinations, edges_to.data(), edges_to.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeLengths, edges_len.data(), edges_len.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeOffsets, edge_offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edgeCount, edge_count.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_populationCities, popCity.data(), P * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_populationPrime, popP.data(), P * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_populationElderly, popE.data(), P * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_shelterLocations, shelterCity.data(), S * sizeof(int), cudaMemcpyHostToDevice));

        
        // Launch random movement kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (P + threadsPerBlock - 1) / threadsPerBlock;
        
        unsigned int seed = time(NULL);
        
        evacuateKernelRandom<<<blocksPerGrid, threadsPerBlock>>>(
            d_results,
            d_visited,
            seed,
            N, S, P, maxDistElder,
            d_populationCities, d_populationPrime, d_populationElderly,
            d_shelterLocations, d_shelterCapacity,
            d_edgeDestinations, d_edgeLengths, d_edgeOffsets, d_edgeCount
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(results.data(), d_results, P * sizeof(EvacuationResult), 
                              cudaMemcpyDeviceToHost));
        
        // Copy back final shelter capacities
        CUDA_CHECK(cudaMemcpy(updatedShelterCap.data(), d_shelterCapacity, S * sizeof(int), 
                              cudaMemcpyDeviceToHost));
        
        // Clean up device memory
        CUDA_CHECK(cudaFree(d_edgeDestinations));
        CUDA_CHECK(cudaFree(d_edgeLengths));
        CUDA_CHECK(cudaFree(d_edgeOffsets));
        CUDA_CHECK(cudaFree(d_edgeCount));
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_visited));
        CUDA_CHECK(cudaFree(d_populationCities));
        CUDA_CHECK(cudaFree(d_populationPrime));
        CUDA_CHECK(cudaFree(d_populationElderly));
        CUDA_CHECK(cudaFree(d_shelterLocations));
        CUDA_CHECK(cudaFree(d_shelterCapacity));

    }
    
    // Convert results to output format
    vector<vector<int>> allPaths(P);
    vector<vector<array<long long, 3>>> allDrops(P);
    
    int resultIndex = 0;
    while (resultIndex < P) {
        allPaths[resultIndex].resize(results[resultIndex].path_size);
        for (int j = 0; j < results[resultIndex].path_size; j++) {
            allPaths[resultIndex][j] = results[resultIndex].path[j];
        }
        
        allDrops[resultIndex].resize(results[resultIndex].drops_size);
        for (int j = 0; j < results[resultIndex].drops_size; j++) {
            allDrops[resultIndex][j][0] = results[resultIndex].drops[j][0];
            allDrops[resultIndex][j][1] = results[resultIndex].drops[j][1];
            allDrops[resultIndex][j][2] = results[resultIndex].drops[j][2];
        }
        resultIndex++;
    }


    // First, declare and allocate memory for the required variables
    long long *path_size = new long long[P];
    long long **paths = new long long*[P];
    long long *num_drops = new long long[P];
    long long ***drops = new long long**[P];

    // Convert the data from results array to the required format
    for (int i = 0; i < P; i++) {
        // Set path_size for this city
        path_size[i] = results[i].path_size;
        
        // Allocate and copy paths
        paths[i] = new long long[results[i].path_size];
        for (int j = 0; j < results[i].path_size; j++) {
            paths[i][j] = results[i].path[j];
        }
        
        // Set num_drops for this city
        num_drops[i] = results[i].drops_size;
        
        // Allocate and copy drops
        drops[i] = new long long*[results[i].drops_size];
        for (int j = 0; j < results[i].drops_size; j++) {
            drops[i][j] = new long long[3];
            for (int k = 0; k < 3; k++) {
                drops[i][j][k] = results[i].drops[j][k];
            }
        }
    }

    // Now write to the output file
    ofstream outfile(argv[2]); 
    if (!outfile) {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }

    // Write paths
    for (long long i = 0; i < P; i++) {
        long long currentPathSize = path_size[i];
        for (long long j = 0; j < currentPathSize; j++) {
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    // Write drops
    for (long long i = 0; i < P; i++) {
        long long currentDropSize = num_drops[i];
        for (long long j = 0; j < currentDropSize; j++) {
            for (int k = 0; k < 3; k++) {
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }

    // Don't forget to free the memory when you're done
    // This would typically be done at the end of your function or program
    for (int i = 0; i < P; i++) {
        delete[] paths[i];
        
        for (int j = 0; j < num_drops[i]; j++) {
            delete[] drops[i][j];
        }
        delete[] drops[i];
    }
    delete[] path_size;
    delete[] paths;
    delete[] num_drops;
    delete[] drops;
    

    outfile.close();
    

    return 0;
}