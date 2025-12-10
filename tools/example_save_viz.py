"""
Example: How to save visualizations for demos

This script shows how to save play visualizations to files that can be
easily opened for presentations or demos.
"""

from tools.play_viz import visualize_predictions, save_animation
from models.bayes_kinematic import BayesianKinematicModel
import pandas as pd

# Example 1: Save directly from visualize_predictions
def example_save_direct():
    """Save animation directly by passing save_path parameter."""
    
    # Load or create your model
    model = BayesianKinematicModel()
    # ... fit your model ...
    
    # Visualize and save in one call
    ani = visualize_predictions(
        model=model,
        week=1,
        game_id=2023090700,
        play_id=56,
        save_path="demo_play.gif",  # Save as GIF
        save_fps=10,  # Optional: override fps
    )
    
    # Animation is already saved, but you can still use it
    # (Note: figure is closed after saving)


# Example 2: Save animation separately
def example_save_separate():
    """Create animation first, then save it separately."""
    
    model = BayesianKinematicModel()
    # ... fit your model ...
    
    # Create animation without saving
    ani = visualize_predictions(
        model=model,
        week=1,
        game_id=2023090700,
        play_id=56,
        # No save_path - animation is returned
    )
    
    # Save it later
    save_animation(ani, "demo_play.mp4", fps=10)  # Save as MP4


# Example 3: Different formats
def example_different_formats():
    """Save in different formats for different use cases."""
    
    model = BayesianKinematicModel()
    # ... fit your model ...
    
    ani = visualize_predictions(
        model=model,
        week=1,
        game_id=2023090700,
        play_id=56,
    )
    
    # GIF - Good for presentations, easy to view
    save_animation(ani, "demo.gif", fps=10)
    
    # MP4 - Better quality, smaller file size (requires ffmpeg)
    # save_animation(ani, "demo.mp4", fps=10, bitrate=2000)
    
    # HTML - Interactive, can embed in web pages
    # save_animation(ani, "demo.html", fps=10)


if __name__ == "__main__":
    print("This is an example file showing how to save visualizations.")
    print("\nQuick usage:")
    print("  ani = visualize_predictions(model, week, game_id, play_id, save_path='demo.gif')")
    print("\nOr save separately:")
    print("  ani = visualize_predictions(model, week, game_id, play_id)")
    print("  save_animation(ani, 'demo.gif')")

