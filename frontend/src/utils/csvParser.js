import Papa from 'papaparse';

/**
 * Parse a CSV file and return the data as an array of objects
 * @param {File} file - The CSV file to parse
 * @returns {Promise<Array>} - A promise that resolves to an array of objects
 */
export const parseCSVFile = (file) => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      delimiter: ",",
      dynamicTyping: true,
      complete: (results) => {
        if (results.errors.length) {
          reject(results.errors);
        } else {
          resolve(results.data);
        }
      },
      error: (error) => {
        reject(error);
      }
    });
  });
};

// Helper function to format to 3 significant figures
const formatToSigFigs = (num, sigFigs = 3) => {
  if (num === null || num === undefined || num === '' || isNaN(num)) return num;
  
  return Number.parseFloat(num).toPrecision(sigFigs);
};

/**
 * Parse a CSV string and return the data as an array of objects
 * @param {string} csvString - The CSV string to parse
 * @returns {Promise<Array>} - A promise that resolves to an array of objects
 */
export const parseCSVString = (csvString) => {
  return new Promise((resolve, reject) => {
    Papa.parse(csvString, {
      header: true,
      delimiter: ",",
      dynamicTyping: true,
      transform: (value, column) => column==='RANK'?value:formatToSigFigs(value),
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length) {
          reject(results.errors);
        } else {
          resolve(results.data);
        }
      },
      error: (error) => {
        reject(error);
      }
    });
  });
};
/**
 * Fetch a CSV file from a URL and parse it
 * @param {string} url - The URL of the CSV file
 * @returns {Promise<Array>} - A promise that resolves to an array of objects
 */
export const fetchAndParseCSV = async (url) => {
  try {
    const response = await fetch(url);
    const csvString = await response.text();
    return parseCSVString(csvString);
  } catch (error) {
    console.error('Error fetching or parsing CSV:', error);
    throw error;
  }
};

/**
 * Get column names from CSV data
 * @param {Array} data - The parsed CSV data
 * @returns {Array} - An array of column names
 */
export const getColumnNames = (data) => {
  if (!data || !data.length) return [];
  return Object.keys(data[0]);
};

/**
 * Filter columns to display based on configuration
 * @param {Array} columns - All available columns
 * @param {Object} config - Configuration object with column preferences
 * @returns {Array} - Filtered array of columns to display
 */
export const filterColumns = (columns, config) => {
  if (!config || !config.columns) return columns;
  
  // If config.columns is an array, use it directly
  if (Array.isArray(config.columns)) {
    return columns.filter(col => config.columns.includes(col));
  }
  
  // If config.columns is an object with include/exclude properties
  if (config.columns.include) {
    return columns.filter(col => config.columns.include.includes(col));
  }
  
  if (config.columns.exclude) {
    return columns.filter(col => !config.columns.exclude.includes(col));
  }
  
  return columns;
};
