/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TGUI - Texus' Graphical User Interface
// Copyright (C) 2012-2022 Bruno Van de Velde (vdv_b@tgui.eu)
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it freely,
// subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented;
//    you must not claim that you wrote the original software.
//    If you use this software in a product, an acknowledgment
//    in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such,
//    and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef TGUI_BACKEND_RENDER_TARGET_SDL_HPP
#define TGUI_BACKEND_RENDER_TARGET_SDL_HPP

#include <TGUI/BackendRenderTarget.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SDL_Window;
struct SDL_Rect;
typedef int GLint;
typedef unsigned int GLuint;

namespace tgui
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Render target that uses SDL to draw the gui
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class TGUI_API BackendRenderTargetSDL : public BackendRenderTargetBase
    {
    public:

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Constructs the render target and informs it to which window it is bound
        ///
        /// @param window  The window on which the gui should be rendered
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        BackendRenderTargetSDL(SDL_Window* window);


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Destructor
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ~BackendRenderTargetSDL();


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Returns the SDL window on which the gui is being drawn
        ///
        /// @return The SDL window that is used by the gui
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        SDL_Window* getWindow() const;

#ifndef TGUI_REMOVE_DEPRECATED_CODE
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Informs the render target about which part of the window is used for rendering
        ///
        /// @param view     Defines which part of the gui is being shown
        /// @param viewport Defines which part of the window is being rendered to
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        TGUI_DEPRECATED("Use setView(view, viewport, targetSize) instead") void setView(FloatRect view, FloatRect viewport) override;
#endif

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Informs the render target about which part of the window is used for rendering
        ///
        /// @param view        Defines which part of the gui is being shown
        /// @param viewport    Defines which part of the window is being rendered to
        /// @param targetSize  Size of the window
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void setView(FloatRect view, FloatRect viewport, Vector2f targetSize) override;


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Draws the gui and all of its widgets
        ///
        /// @param root  Root container that holds all widgets in the gui
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void drawGui(const std::shared_ptr<RootContainer>& root) override;

#ifndef TGUI_REMOVE_DEPRECATED_CODE
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Adds another clipping region
        ///
        /// @param states  Render states to use for drawing
        /// @param rect  The clipping region
        ///
        /// If multiple clipping regions were added then contents is only shown in the intersection of all regions.
        ///
        /// @warning Every call to addClippingLayer must have a matching call to removeClippingLayer.
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void addClippingLayer(const RenderStates& states, FloatRect rect) override;


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Removes the last added clipping region
        ///
        /// @warning The addClippingLayer function must have been called before calling this function.
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void removeClippingLayer() override;
#endif

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Draws a texture
        ///
        /// @param states  Render states to use for drawing
        /// @param sprite  Image to draw
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void drawSprite(const RenderStates& states, const Sprite& sprite) override;


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Draws some text
        ///
        /// @param states  Render states to use for drawing
        /// @param text    Text to draw
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void drawText(const RenderStates& states, const Text& text) override;


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Draws one or more triangles (using the color that is specified in the vertices)
        ///
        /// @param states       Render states to use for drawing
        /// @param vertices     Pointer to first element in array of vertices
        /// @param vertexCount  Amount of elements in the vertex array
        /// @param indices      Pointer to first element in array of indices
        /// @param indexCount   Amount of elements in the indices array
        ///
        /// If indices is a nullptr then vertexCount must be a multiple of 3 and each set of 3 vertices will be seen as a triangle.
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void drawTriangles(const RenderStates& states, const Vertex* vertices, std::size_t vertexCount, const int* indices = nullptr, std::size_t indexCount = 0) override;


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    protected:

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @brief Called from addClippingLayer and removeClippingLayer to apply the clipping
        ///
        /// @param clipRect      View rectangle to apply
        /// @param clipViewport  Viewport to apply
        ///
        /// Both rectangles may be empty when nothing that will be drawn is going to be visible.
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void updateClipping(FloatRect clipRect, FloatRect clipViewport) override;


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Calls glEnableVertexAttribArray and glVertexAttribPointer.
        // Called only once when using a VAO (OpenGL or GLES 3.x), but on every draw when using GLES 2.0
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void setVertexAttribs();


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Creates the vertex and index buffers
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void createBuffers();


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Fills the vertex and index buffers
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void prepareVerticesAndIndices(const Sprite& sprite);


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Fills the vertex and index buffers
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void prepareVerticesAndIndices(const Vertex* vertices, std::size_t vertexCount, const int* indices, std::size_t indexCount, Vector2u textureSize = {1,1});


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Changes the bound texture if another texture was currently set
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void changeTexture(GLuint textureId, bool force = false);


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Updates the matrix in the shader that will transform each vertex
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void updateTransformation(const Transform& transform);


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Draws a single line of text
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void drawTextLine(const SDL_Rect& bounding, GLuint textureId);


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    protected:

        GLuint m_shaderProgram = 0;
        GLuint m_vertexArray = 0;
        GLuint m_vertexBuffer = 0;
        GLuint m_indexBuffer = 0;
        GLuint m_emptyTexture = 0;
        std::size_t m_vertexBufferSize = 0;
        std::size_t m_indexBufferSize = 0;

        SDL_Window* m_window = nullptr;

#ifndef TGUI_REMOVE_DEPRECATED_CODE
        TGUI_DEPRECATED("Use m_targetSize instead") int m_windowWidth = 0;
        TGUI_DEPRECATED("Use m_targetSize instead") int m_windowHeight = 0;
        TGUI_DEPRECATED("Calculate by using m_viewport and m_targetSize instead") std::array<int, 4> m_viewportGL = {};
        TGUI_DEPRECATED("Use m_clipLayers instead") std::vector<std::pair<FloatRect, std::array<int, 4>>> m_clippingLayers;
#endif

        Transform m_projectionTransform;
        GLint m_projectionMatrixShaderUniformLocation = 0;
        GLint m_positionShaderLocation = 0;
        GLint m_colorShaderLocation = 1;
        GLint m_texCoordShaderLocation = 2;
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // TGUI_BACKEND_RENDER_TARGET_SDL_HPP
