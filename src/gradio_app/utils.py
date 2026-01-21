import os
import pandas as pd

from datetime import datetime
from typing import Union

def export_to_csv(data:pd.DataFrame) -> Union[str, None]:
    """
    Exports the detection results to a CSV file.

    Args:
        data (pd.DataFrame): The detection results to export as a CSV.
    """
    if data is None or data.empty:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_results_{timestamp}.csv"
    
    temp_path = os.path.join(os.getcwd(), filename)
    data.to_csv(temp_path, index=False)
    
    return temp_path


def cleanup_temp_files():
    """
    Helper function for cleaning up temporary files.
    """
    for file in os.listdir():
        if file.startswith("detection_results_") and file.endswith(".csv"):
            try:
                os.remove(file)
            except:
                pass