#include <SFML/Graphics.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <TGUI/TGUI.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
//
//using json = nlohmann::json;
//
//enum class GameState {
//    MENU,
//    PLAY,
//    LOAD,
//    SETTINGS,
//    ERROR
//};
//
//// Default config values
//json default_config = {
//        {"resolution_option", 0},
//        {"fullscreen", false},
//        {"fps", 60},
//};
//
//// Resolution options
//std::vector<sf::Vector2u> RESOLUTION_OPTIONS = {
//        sf::Vector2u(1280, 720),
//        sf::Vector2u(1600, 900),
//        sf::Vector2u(1920, 1080),
//        sf::Vector2u(2560, 1440),
//        sf::Vector2u(3840, 2160)
//};
//
//// The current game state
//GameState current_state = GameState::MENU;
//std::string message;
//
//// Font
//sf::Font PIXEL_FONT;
//
//// Colors
//sf::Color BACKGROUND_COLOR(87, 135, 207);
//sf::Color BUTTON_COLOR(60, 129, 207);
//sf::Color TEXT_COLOR(197, 209, 237);
//
//// Load the configuration from the config.json file
//json load_config() {
//    std::ifstream configFile("config.json");
//    if (!configFile) {
//        std::cout << "ERROR: Config does not exist, using defaults." << std::endl;
//        return nullptr;
//    }
//
//    json new_config;
//    configFile >> new_config;
//    return new_config;
//}
//
//// Save the configuration to the config.json file
//void save_config(const json& new_config) {
//    std::ofstream configFile("config.json");
//    configFile << new_config.dump(4);
//}
//
//void menu(sf::RenderWindow& window, tgui::GuiSFML& gui) {
//    message = "Created by Collin Miller";
//    //auto message_time = clock.getElapsedTime().asMilliseconds() + 5000;
//
//    // Instead of creating Rectangles, we create buttons directly
//    auto new_game_button = tgui::Button::create();
//    new_game_button->setSize({"50%", "12.5%"});
//    new_game_button->setPosition({"25%", "25%"});
//    new_game_button->setText("New Game"); // Set the text of the button
//    new_game_button->onPress([](){ current_state = GameState::PLAY; message = ""; });
//    gui.add(new_game_button);
//
//    auto load_game_button = tgui::Button::create();
//    load_game_button->setSize({"50%", "12.5%"});
//    load_game_button->setPosition({"25%", "40%"});
//    load_game_button->setText("Load Game");
//    load_game_button->onPress([](){ current_state = GameState::LOAD; message = ""; });
//    gui.add(load_game_button);
//
//    auto settings_button = tgui::Button::create();
//    settings_button->setSize({"50%", "12.5%"});
//    settings_button->setPosition({"25%", "55%"});
//    settings_button->setText("Settings");
//    settings_button->onPress([](){ current_state = GameState::SETTINGS; message = ""; });
//    gui.add(settings_button);
//
//    auto exit_button = tgui::Button::create();
//    exit_button->setSize({"50%", "12.5%"});
//    exit_button->setPosition({"25%", "70%"});
//    exit_button->setText("Exit");
//    exit_button->setTextSize(48);
//    exit_button->onPress([&](){ window.close(); });
//    gui.add(exit_button);
//
//    // Handle events
//    sf::Event event;
//    while (window.pollEvent(event)) {
//        if (event.type == sf::Event::Closed)
//            window.close();
//
//        gui.handleEvent(event);
//    }
//
//    window.clear();
//    gui.draw();
//    window.display();
//}
//
//int main() {
//
//    // Load the configuration from the config.json file
//    std::ifstream configFile("config.json");
//    json config = load_config();
//    if (config.is_null()) { // Use default values
//        config = default_config;
//        save_config(default_config);
//    }
//
//    sf::RenderWindow window;
//    if (config["fullscreen"].get<bool>()) {
//        window.create(sf::VideoMode::getDesktopMode(), "Squares", sf::Style::Fullscreen);
//    } else {
//        window.create(sf::VideoMode::getDesktopMode(), "Squares", sf::Style::Default);
//    }
//
//    tgui::GuiSFML gui(window);
//    sf::Clock clock;
//
//    while (window.isOpen()) {
//        sf::Event event;
//        while (window.pollEvent(event)) {
//            if (event.type == sf::Event::Closed) {
//                window.close();
//            }
//        }
//
//        switch (current_state) {
//            case GameState::MENU:
//                menu(window, gui);
//                break;
//            case GameState::PLAY:
//                //play();
//                break;
//            case GameState::LOAD:
//                //load();
//                break;
//            case GameState::SETTINGS:
//                //settings();
//                break;
//            case GameState::ERROR:
//                //error();
//                break;
//        }
//
//        // Cap the framerate
//        sf::sleep(sf::seconds(1.f / config["fps"].get<int8_t>()));
//    }
//
//    return 0;
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  MAIN
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MapGenerator.h"

// Main
int main() {
    // Generate maps
    MapConfig mapConfig = loadMapConfig();
    MapData mapData = generate(mapConfig);
    saveColorAndHeightMaps(mapData);

    return 0;
}
