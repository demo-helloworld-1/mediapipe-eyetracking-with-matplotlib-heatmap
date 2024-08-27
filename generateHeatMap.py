import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmap(csv_file, output_image='heatmap.png'):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if the necessary columns exist
    if 'x_screen' not in df.columns or 'y_screen' not in df.columns:
        raise ValueError("CSV file must contain 'x_screen' and 'y_screen' columns.")
    
    # Create a pivot table for heatmap generation
    heatmap_data = pd.pivot_table(df, values=None, index='y_screen', columns='x_screen', aggfunc='size', fill_value=0)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=True)
    plt.title('Heatmap of Eye Tracking Coordinates')
    plt.xlabel('Screen X Coordinate')
    plt.ylabel('Screen Y Coordinate')
    
    # Save the heatmap as an image file
    plt.savefig(output_image)
    plt.show()

# Example usage
generate_heatmap('coordinates.csv', 'heatmap.png')
