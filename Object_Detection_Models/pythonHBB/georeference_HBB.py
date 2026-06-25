import json
from pathlib import Path

# Helper function to parse TWF files
def parse_twf(twf_path):
    '''Parses a .twf and returns its georeferencing parameters.

            Parameters: A twf_path: Path to the .twf file.

            Returns: A tuple (A, B, C, D, E, F) representing the georeferencing parameters.
    '''
    with open(twf_path, 'r') as f:
        lines = f.readlines()
    # TWF format:
    # A (pixel width)
    # B (rotation around Y, usually 0)
    # C (rotation around X, usually 0)
    # D (pixel height, usually negative)
    # E (X coordinate of center of upper-left pixel)
    # F (Y coordinate of center of upper-left pixel)
    A = float(lines[0].strip())
    B = float(lines[1].strip())
    C = float(lines[2].strip())
    D = float(lines[3].strip())
    E = float(lines[4].strip())
    F = float(lines[5].strip())
    return A, B, C, D, E, F

# Helper function to convert pixel coordinates to geographic coordinates
def pixel_to_geo(px, py, A, B, C, D, E, F):
    '''Converts pixel coordinates (px, py) to geographic coordinates (lon, lat) using TWF parameters.
    '''
    # X_geo = A * px + B * py + E
    # Y_geo = C * px + D * py + F
    
    # Standard world file convention uses the center of the pixel, so often +0.5 is applied to pixel coords.
    # However, for bounding box corners, we often just use the direct pixel values.
    # Given B and C are typically 0 for non-rotated images, the formulas simplify.
    lon = A * px + B * py + E
    lat = C * px + D * py + F
    return lon, lat

def georeference_HBB (predict_results, tif_dir, geojson_output_path):
    '''Converts bounding box predictions from pixel coordinates to geographic coordinates using TWF files and saves them as a GeoJSON file.

            Parameters: A Prediction results from the YOLO model.
                        B Directory containing the original .tif images and their corresponding .twf files.
                        C Path to save the output GeoJSON file.

            Returns: None 
    '''
    geojson_output_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_features = []

    for result in predict_results:
        # Contains the path to the predicted .png image.
        png_image_path = Path(result.path)
        image_filename_stem = png_image_path.stem 

        # Paths to the original .tif and .twf files
        tif_file_path = tif_dir / f"{image_filename_stem}.tif"
        twf_file_path = tif_dir / f"{image_filename_stem}.tfw"

        if not twf_file_path.exists():
            print(f"Warning: TWF file not found for {image_filename_stem}. Skipping georeferencing for this image.")
            continue

        # Parse the TWF file
        A, B, C, D, E, F = parse_twf(twf_file_path)

        # Iterate through each detected bounding box
        for *xyxy, conf, cls in result.boxes.data.tolist():
            x1, y1, x2, y2 = xyxy
            
            # Convert corners to geographic coordinates
            lon1, lat1 = pixel_to_geo(x1, y1, A, B, C, D, E, F)
            lon2, lat2 = pixel_to_geo(x2, y1, A, B, C, D, E, F) 
            lon3, lat3 = pixel_to_geo(x2, y2, A, B, C, D, E, F)
            lon4, lat4 = pixel_to_geo(x1, y2, A, B, C, D, E, F) 

            # Create a GeoJSON Polygon feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon1, lat1],
                        [lon2, lat2],
                        [lon3, lat3],
                        [lon4, lat4],
                        [lon1, lat1]  
                    ]]
                },
                "properties": {
                    "confidence": conf,
                    "class": result.names[int(cls)],
                    "image_source": str(tif_file_path),
                    "pixel_bbox": [x1, y1, x2, y2]
                }
            }
            geojson_features.append(feature)

    # Create a GeoJSON FeatureCollection
    geoj_collection = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # Save the GeoJSON to a file
    with open(geojson_output_path, 'w') as f:
        json.dump(geoj_collection, f, indent=2)

    print(f"Successfully saved {len(geojson_features)} bounding box predictions to {geojson_output_path}")