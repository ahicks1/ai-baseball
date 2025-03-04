const fs = require('fs');
const path = require('path');

// Create directories if they don't exist
const ensureDirectoryExists = (dirPath) => {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`Created directory: ${dirPath}`);
  }
};

// Copy file from source to destination
const copyFile = (source, destination) => {
  try {
    fs.copyFileSync(source, destination);
    console.log(`Copied: ${source} -> ${destination}`);
  } catch (error) {
    console.error(`Error copying ${source}: ${error.message}`);
  }
};

// Main function to copy data files
const copyDataFiles = () => {
  // Source and destination directories
  const sourceDir = path.resolve(__dirname, '../data/rankings_with_scarcity');
  const destDir = path.resolve(__dirname, 'public/data/rankings_with_scarcity');
  
  // Ensure destination directory exists
  ensureDirectoryExists(destDir);
  
  // Files to copy
  const filesToCopy = [
    'batting_rankings.csv',
    'pitching_rankings.csv',
    'overall_rankings.csv'
  ];
  
  // Copy each file
  filesToCopy.forEach(file => {
    const sourcePath = path.join(sourceDir, file);
    const destPath = path.join(destDir, file);
    
    if (fs.existsSync(sourcePath)) {
      copyFile(sourcePath, destPath);
    } else {
      console.warn(`Source file not found: ${sourcePath}`);
    }
  });
  
  console.log('Data files copied successfully!');
};

// Run the copy function
copyDataFiles();
