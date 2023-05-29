#include <vector>
#include <cmath>

#ifndef SQUARES_MAP_GENERATOR_H
#define SQUARES_MAP_GENERATOR_H

const int DEFAULT_MAP_WIDTH = 1000, DEFAULT_MAP_HEIGHT = 1000;
const float DEFAULT_SEA_LEVEL = 0.4f, DEFAULT_RIVER_DENSITY = 0.00001f;

// A struct to store color
struct MapColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// Struct that stores map configuration
struct MapConfig {
    //// Map general config ////
    // Map base parameters
    int channels = 3; // RGB channels
    int mapWidth = 7000;
    int mapHeight = 3500;
    float seaLevel = 0.4f; // Dictates where water ends and land starts (must be between 0.15 - 0.65)!!!

    //// Noise generation config ////
    // Noise generation
    float frequency = 0.03; // Default: 0.25
    int octaves = 30; // Default: 12
    float lacunarity = 2.0f; // Default: 2.0
    float persistence = 0.61f; // Default: 0.5
    float distanceFromCenter = 0.8f; // Default: 0.8
    // Point generation
    int pointRadius = 450;
    int edgeClearance = 500; // 500
    float steepness = 2.0f;
    float baseScalingFactor = 0.32f;
    float maxScalingFactor = 1.0f;
    float maxPointHeight = 1.0f;
    // Cluster generation
    int numClusters = 6; // 6
    int clusterMinPoints = 15;
    int clusterMaxPoints = 75;
    int clusterDensityRadius = 300;
    // Mountain point generation
    int mountainPointRadius = 150;
    float mountainSteepness = 1.0f;
    float baseMountainScalingFactor = 1.0f;
    float maxMountainScalingFactor = 1.5f;
    float maxMountainPointHeight = 1.0f;
    // Curve generation
    int numCurves = 10;
    int curveMinPoints = 20;
    int curveMaxPoints = 40;
    int curveDeviation = 15;
    int minAxis = 100;
    int maxAxis = 600;
    int curveMinAngle = 90;
    int curveMaxAngle = 270;

    //// River generation config ////
    // River spawning
    float riverDensity = 0.0001f; // Default: 0.0001
//    int riverCount = riverDensity * (mapWidth * mapHeight); // Should be proportional to map size
    int riverCount = 200;
    float maxRiverSpawnHeight = seaLevel + 0.35;
    float minRiverSpawnHeight = seaLevel + 0.08;
    // River path finding
    int circularSearchNextPathMinDistance = 20; // Default: 20
    float minRiverDespawnHeight = seaLevel - 0.05;
    // River random point placement
    int randomPointPlacementSpacing = 20; // Default: 10
    float maxRandomPointDeviation = 40.0; // Default: 15.0
    float minRandomPointDeviation = 15.0; // Default: 3.0
    // River filtering
    int maxRiverStraightness = 15; // Default: 15
    int minRiverLength = 0; // Default: 10 (This can lead to rivers that go nowhere)
    // River loop cutting
    int loopClearanceWindow = 5;
    int loopIntersectionRange = 3;
    // River intersection stitching
    int riverIntersectionRange = 3; // Default: 10
    // River carving
    float riverStartRadius = 1.0f; // Default: 1.0
    float riverMaxRadius = 5.0f; // Default: 5.0
    float riverRadiusLengthScale = 1.5f; // Default: 1.5
    float terrainCarveDistortion = 0.025f; // Default: 0.025
    float terrainCarveStartRadius = 3.0f; // Default: 1.5
    float terrainCarveMaxRadius = 8.0f; // Default: 8.0
    float terrainCarveLengthScale = 1.5f; // Default: 1.25
    int maxRiverRadiusLength = 800; // Default: 300
    float riverMinDepth = seaLevel - seaLevel * 0.03;
    float riverMaxDepth = seaLevel - seaLevel * 0.1;
    float terrainMinDepth = seaLevel;

    //// Erode heightmap parameters ////
    float dropDensity = 0.333;
    int numDrops = dropDensity * (mapWidth * mapHeight); // Should be proportional to map size
    float minDropSpawnHeight = seaLevel + (1.0 - seaLevel) * 0.17f;
    int erosionRadius = 3; // Default: 3
    float inertia = 0.05f; // Default: 0.05
    float sedimentCapacityFactor = 4.0f; // Default: 4.0
    float minSedimentCapacity = 0.01f; // Default: 0.01
    float erodeSpeed = 0.3f; // Default: 0.3
    float depositSpeed = 0.3f; // Default: 0.3
    float evaporateSpeed = 0.01f; // Default: 0.01
    float gravity = 4.0f; // Default: 4.0
    int maxDropLifeTime = 300; // Default: 30
    float initialWaterVolume = 1.0f; // Default: 1.0
    float minWaterVolume = 0.01f; // Default 0.01
    float initialSpeed = 1.0f; // Default: 1.0
    float minDropDespawnHeight = seaLevel - seaLevel * 0.13;

    //// Paint colormap parameters ////
    // Where on the heightmap each color level ends
    float waterLevel = seaLevel;
    float beachLevel = seaLevel + 0.001;
    float grassLevel = seaLevel + 0.35;
    float mountainLevel = seaLevel + 0.45;

    // Number of layers for each level for the color interpolation
    int waterLayers = 15;
    int beachLayers = 1;
    int grassLayers = 15;
    int mountainLayers = 6;
    int snowLayers = 3;

    // Maximum and minimum RGB colors for each level
    MapColor minWaterColor = {37, 89, 134};
    MapColor maxWaterColor = {140, 183, 220};

    MapColor minBeachColor = {230, 218, 166};
    MapColor maxBeachColor = {206, 190, 124};

    MapColor minGrassColor = {180, 218, 91};
    MapColor maxGrassColor = {93, 121, 30};

    MapColor minMountainColor = {92, 88, 70};
    MapColor maxMountainColor = {144, 142, 136};

    MapColor minSnowColor = {223, 222, 216};
    MapColor maxSnowColor = {245, 244, 239};

    // Constructor
    MapConfig(int _mapWidth, int _mapHeight, float _seaLevel, float _riverDensity) : mapWidth(_mapWidth), mapHeight(_mapHeight), seaLevel(_seaLevel), riverDensity(_riverDensity) {}

    // Constructor
    MapConfig() {}
};

// Struct that stores map data
struct MapData {
    MapConfig mapConfig;
    std::vector<float> heightMap;  // Store the world map height map
    std::vector<unsigned char> grayscaleMap; // Store the world map with color
    std::vector<unsigned char> colorMap; // Store the world map with color
};

// Load the map configuration from MapConfig.json
MapConfig loadMapConfig();

// Generate map data with a map config
MapData generate(MapConfig mapConfig);

// Save color and height maps from map data
bool saveColorAndHeightMaps(MapData mapData);

#endif
