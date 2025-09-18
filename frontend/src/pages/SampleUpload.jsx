import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import { CloudUpload as UploadIcon } from '@mui/icons-material';
import axios from 'axios';

function SampleUpload() {
  const [files, setFiles] = useState([]);
  const [metadata, setMetadata] = useState({
    collection_date: '',
    location: {
      latitude: '',
      longitude: '',
      depth: '',
    },
    environmental_data: {
      temperature: '',
      salinity: '',
    },
    collection_method: '',
    project_id: '',
  });
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    setFiles(acceptedFiles);
    setError(null);
    setSuccess(false);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt', '.fasta', '.fastq'],
    },
    maxFiles: 1,
  });

  const handleMetadataChange = (field, value) => {
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      setMetadata((prev) => ({
        ...prev,
        [parent]: {
          ...prev[parent],
          [child]: value,
        },
      }));
    } else {
      setMetadata((prev) => ({
        ...prev,
        [field]: value,
      }));
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select a file to upload');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', files[0]);
    formData.append('metadata', JSON.stringify(metadata));

    try {
      const response = await axios.post('/api/v1/samples/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setSuccess(true);
      setFiles([]);
      setMetadata({
        collection_date: '',
        location: {
          latitude: '',
          longitude: '',
          depth: '',
        },
        environmental_data: {
          temperature: '',
          salinity: '',
        },
        collection_method: '',
        project_id: '',
      });
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during upload');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', py: 4 }}>
      <Typography variant="h4" gutterBottom>
        Upload Sample
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Sample uploaded successfully!
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper
            {...getRootProps()}
            sx={{
              p: 3,
              textAlign: 'center',
              cursor: 'pointer',
              bgcolor: isDragActive ? 'action.hover' : 'background.paper',
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
            }}
          >
            <input {...getInputProps()} />
            <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive
                ? 'Drop the files here...'
                : 'Drag & drop files here, or click to select files'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Supported formats: .txt, .fasta, .fastq
            </Typography>
            {files.length > 0 && (
              <Typography variant="body1" sx={{ mt: 2 }}>
                Selected file: {files[0].name}
              </Typography>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sample Metadata
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Collection Date"
                    type="datetime-local"
                    value={metadata.collection_date}
                    onChange={(e) =>
                      handleMetadataChange('collection_date', e.target.value)
                    }
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Latitude"
                    type="number"
                    value={metadata.location.latitude}
                    onChange={(e) =>
                      handleMetadataChange('location.latitude', e.target.value)
                    }
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Longitude"
                    type="number"
                    value={metadata.location.longitude}
                    onChange={(e) =>
                      handleMetadataChange('location.longitude', e.target.value)
                    }
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Depth (m)"
                    type="number"
                    value={metadata.location.depth}
                    onChange={(e) =>
                      handleMetadataChange('location.depth', e.target.value)
                    }
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Temperature (°C)"
                    type="number"
                    value={metadata.environmental_data.temperature}
                    onChange={(e) =>
                      handleMetadataChange(
                        'environmental_data.temperature',
                        e.target.value
                      )
                    }
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Salinity (PSU)"
                    type="number"
                    value={metadata.environmental_data.salinity}
                    onChange={(e) =>
                      handleMetadataChange(
                        'environmental_data.salinity',
                        e.target.value
                      )
                    }
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Collection Method"
                    value={metadata.collection_method}
                    onChange={(e) =>
                      handleMetadataChange('collection_method', e.target.value)
                    }
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Project ID"
                    value={metadata.project_id}
                    onChange={(e) =>
                      handleMetadataChange('project_id', e.target.value)
                    }
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={uploading || files.length === 0}
            startIcon={uploading ? <CircularProgress size={20} /> : <UploadIcon />}
            fullWidth
          >
            {uploading ? 'Uploading...' : 'Upload Sample'}
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
}

export default SampleUpload;
