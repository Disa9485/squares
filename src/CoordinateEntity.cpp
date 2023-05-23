#include <SFML/Graphics.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include "CoordinateEntity.hpp"

void Grid::add_entity(CoordinateEntity* entity) {
    int x = static_cast<int>(entity->current_x_coordinate / cell_size);
    int y = static_cast<int>(entity->current_y_coordinate / cell_size);
    cells[{x, y}].push_back(entity);
}

void Grid::remove_entity(CoordinateEntity* entity) {
    int x = static_cast<int>(entity->current_x_coordinate / cell_size);
    int y = static_cast<int>(entity->current_y_coordinate / cell_size);
    cells[{x, y}].erase(std::remove(cells[{x, y}].begin(), cells[{x, y}].end(), entity), cells[{x, y}].end());
}

std::vector<CoordinateEntity*> Grid::get_nearby_entities(CoordinateEntity* entity) {
    int x = static_cast<int>(entity->current_x_coordinate / cell_size);
    int y = static_cast<int>(entity->current_y_coordinate / cell_size);
    std::vector<CoordinateEntity*> nearby_entities;
    for (int i = x-1; i <= x+1; ++i) {
        for (int j = y-1; j <= y+1; ++j) {
            auto it = cells.find({i, j});
            if (it != cells.end()) {
                nearby_entities.insert(nearby_entities.end(), it->second.begin(), it->second.end());
            }
        }
    }
    return nearby_entities;
}

CoordinateEntity::CoordinateEntity(int x_size, int y_size, float boundary, Grid* grid, float start_x_coordinate, float start_y_coordinate, sf::Color color)
        : x_size(x_size), y_size(y_size), boundary(boundary), grid(grid), current_x_coordinate(start_x_coordinate), current_y_coordinate(start_y_coordinate), color(color) {
    target_x_coordinate = current_x_coordinate;
    target_y_coordinate = current_y_coordinate;
    angle = 0;
    grid->add_entity(this);
}

void CoordinateEntity::draw(sf::RenderWindow& window) {
    float half_width = x_size / 2;
    float half_height = y_size / 2;

    sf::ConvexShape polygon;
    polygon.setPointCount(4);

    std::vector<sf::Vector2f> corners = {
            sf::Vector2f(-half_width, -half_height),
            sf::Vector2f(half_width, -half_height),
            sf::Vector2f(half_width, half_height),
            sf::Vector2f(-half_width, half_height)
    };

    float angle_rad = angle * M_PI / 180.0f;
    float cos_angle = cos(angle_rad);
    float sin_angle = sin(angle_rad);
    for (int i = 0; i < 4; ++i) {
        float x = corners[i].x * cos_angle - corners[i].y * sin_angle + current_x_coordinate;
        float y = corners[i].x * sin_angle + corners[i].y * cos_angle + current_y_coordinate;
        polygon.setPoint(i, sf::Vector2f(x, y));
    }

    polygon.setFillColor(color);

    window.draw(polygon);
}

void CoordinateEntity::update() {
    float direction = atan2(target_y_coordinate - current_y_coordinate, target_x_coordinate - current_x_coordinate);
    direction = direction * 180.0f / M_PI;
    direction = fmod(direction + 180.0f, 360.0f) - 180.0f;

    float difference = direction - angle;
    difference = fmod(difference + 180.0f, 360.0f) - 180.0f;

    if (difference > 0) {
        angle += std::min(5.0f, difference);
    } else {
        angle -= std::min(5.0f, -difference);
    }

    float dx = target_x_coordinate - current_x_coordinate;
    float dy = target_y_coordinate - current_y_coordinate;
    float distance = hypot(dx, dy);

    if (distance > 1) {
        dx /= distance;
        dy /= distance;

        for (CoordinateEntity* entity : grid->get_nearby_entities(this)) {
            if (entity != this) {
                float diff_x = current_x_coordinate - entity->current_x_coordinate;
                float diff_y = current_y_coordinate - entity->current_y_coordinate;
                float distance_to_entity = hypot(diff_x, diff_y);

                if (distance_to_entity < (boundary + entity->boundary)) {
                    if (distance_to_entity == 0) {
                        continue;
                    }

                    float avoidance_x = diff_x / distance_to_entity;
                    float avoidance_y = diff_y / distance_to_entity;

                    float avoidance = (1 - distance_to_entity / (boundary + entity->boundary));

                    dx += avoidance_x * avoidance;
                    dy += avoidance_y * avoidance;
                }
            }
        }

        current_x_coordinate += dx;
        current_y_coordinate += dy;

        int old_cell_x = static_cast<int>(current_x_coordinate / grid->cell_size);
        int old_cell_y = static_cast<int>(current_y_coordinate / grid->cell_size);
        int new_cell_x = static_cast<int>((current_x_coordinate + dx) / grid->cell_size);
        int new_cell_y = static_cast<int>((current_y_coordinate + dy) / grid->cell_size);

        if (old_cell_x != new_cell_x || old_cell_y != new_cell_y) {
            grid->remove_entity(this);
            grid->add_entity(this);
        }
    }
}

void CoordinateEntity::set_target(float target_x_coordinate, float target_y_coordinate) {
    this->target_x_coordinate = target_x_coordinate;
    this->target_y_coordinate = target_y_coordinate;
}
