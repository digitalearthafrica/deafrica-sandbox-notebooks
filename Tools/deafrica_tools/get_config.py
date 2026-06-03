# The config dictionary for Digital Earth Africa and Digital Earth Australia products 
# is determined by the product's definition available at the DE Africa Metadata Explorer
# DE Australia Metadata Explorer respectively. 
# For example the DE Africa Sentinel 2 product definition YAML file is available 
# at https://explorer.digitalearth.africa/products/s2_l2a.odc-product.yaml
# The DE Australia Sentinel-2A Definitive product definition YAML file is available 
# at https://explorer.sandbox.dea.ga.gov.au/products/ga_s2am_ard_provisional_3#definition-doc


# Import the required packages.
import requests
import yaml 
from yaml.loader import SafeLoader


def check_product_name_validity(product_name, product_list_url, product_source, explorer_url):
    
    """
    Helper function that checks if the product name passed to the product_name argument of the 
    `get_product_config(...)` function is a valid Digital Earth Africa or 
    Digital Earth Australia product name. 
    
    Raises an error if the product_name is invalid. 
    
    """
    
    product_list_resp = requests.get(product_list_url)
    product_list_resp_text = product_list_resp.text
    product_list = product_list_resp_text.split("\n")

    if product_name in product_list:
        pass
    else: 
        raise ValueError(f"Invalid product name. Please see the {product_source} Metadata Explorer: {explorer_url} for a list of valid product names.") 
    
    
def get_product_config(product_name, profile="deafrica"):

    """
    Downloads the product definition YAML file and builds a dictionary containing
    the pixel data type, nodata value, unit attribute and band aliases 
    of the product. The dictionary can be passed to the `odc.stac.load` 
    `stac_cfg` parameter.  

    Last modified: April 2022

    Parameters
    ----------

    product_name : str
                   The Digital Earth Africa or Digital Earth Australia  product name. 

    profile : str
              Defines whether the product is a Digital Earth Africa (profile="deafrica") 
              or a Digital Earth Australia product (profile="deaustralia").
              The default value is profile="deafrica".
              
                        
    Returns
    -------
    config : dict
             A dictionary containing the product's pixel data type, nodata 
             value used, unit attribute and band aliases.
    """
	
    # Define the url of the product's product definition YAML file.
    if profile == "deafrica":
        product_list_url = "https://explorer.digitalearth.africa/products.txt"
        product_source = "Digital Earth Africa"
        explorer_url = "https://explorer.digitalearth.africa/products"
        product_definition_yaml_url = f"https://explorer.digitalearth.africa/products/{product_name}.odc-product.yaml"
    elif profile == "deaustralia":
        product_list_url = "https://explorer.sandbox.dea.ga.gov.au/products.txt"
        product_source = "Digital Earth Australia"
        explorer_url = "https://explorer.sandbox.dea.ga.gov.au/products"
        product_definition_yaml_url = f"https://explorer.sandbox.dea.ga.gov.au/products/{product_name}.odc-product.yaml"
    else:
        raise ValueError(f"Invalid profile. Please ensure the profile entered is either 'deafrica' or 'deaustralia', to specify whether the product is a Digital Earth Africa or Digital Earth Australia product.")  
    
    # Check the validity of the product name provided. 
    check_product_name_validity(product_name, product_list_url, product_source, explorer_url)
        
    # Download the product definition yaml file from the Metadata Explorer.
    resp = requests.get(product_definition_yaml_url)
    resp_text = resp.text
    # Load the product definition YAML file as a dictionary. 
    product_definition = yaml.load(resp_text, Loader=SafeLoader)
    
    # Get the product measurements metadata from the product definition.
    product_definition_measurements = product_definition['measurements']
    
    # Get the 'dtype', 'units' and 'nodata' information for each band in the collection.
    assets_dict = {}
    for measurement in product_definition_measurements:
        assets_dict[measurement['name']] = {'data_type': measurement['dtype'],
                                               'unit': measurement['units'],
                                               'nodata': measurement['nodata']}

    # Get the aliases for each band.
    aliases_dict = {}
    for measurement in product_definition_measurements:
        if 'aliases' in measurement: 
            for alias in measurement['aliases']:
                aliases_dict[alias] = measurement['name']
    
    # Get the 'dtype', 'units', 'nodata' and 'aliases' information for each band in the collection as a dictionary.
    if len(aliases_dict) == 0:
        config = {product_name: {"assets": assets_dict}}
    else: 
        config = {product_name: {"assets": assets_dict,
                            "aliases": aliases_dict}}
    
    # Output the config dictionary.
    return config
