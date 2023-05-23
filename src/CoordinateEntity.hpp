#include <SFML/Graphics.hpp>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>

class CoordinateEntity; // forward declaration

class Grid {
public:
    Grid(int width, int height, int cell_size) : width(width), height(height), cell_size(cell_size) {}

    void add_entity(CoordinateEntity* entity);

    void remove_entity(CoordinateEntity* entity);

    std::vector<CoordinateEntity*> get_nearby_entities(CoordinateEntity* entity);

    int cell_size;

private:
    int width, height;
    std::map<std::pair<int, int>, std::vector<CoordinateEntity*>> cells;
};

class CoordinateEntity {
public:
    CoordinateEntity(int x_size, int y_size, float boundary, Grid* grid, float start_x_coordinate, float start_y_coordinate, sf::Color color);

    void draw(sf::RenderWindow& window);

    void update();

    void set_target(float target_x_coordinate, float target_y_coordinate);

    float current_x_coordinate;
    float current_y_coordinate;

    // Make these accessible for Grid
    float x_size, y_size, boundary;
    Grid* grid;

private:
    sf::Color color;

    float target_x_coordinate;
    float target_y_coordinate;

    float angle;
};