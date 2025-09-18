import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  AccountTree as AccountTreeIcon,
} from '@mui/icons-material';
import axios from 'axios';

const TaxonomyNode = ({ node, level = 0 }) => {
  const [expanded, setExpanded] = useState(false);
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleExpand = async () => {
    if (!expanded && children.length === 0) {
      setLoading(true);
      try {
        const response = await axios.get(`/api/taxonomy/${node.id}/children`);
        setChildren(response.data);
      } catch (error) {
        console.error('Error fetching taxonomy children:', error);
      }
      setLoading(false);
    }
    setExpanded(!expanded);
  };

  return (
    <>
      <ListItem
        button
        onClick={handleExpand}
        sx={{ pl: level * 2 }}
      >
        <ListItemIcon>
          <AccountTreeIcon />
        </ListItemIcon>
        <ListItemText
          primary={node.name}
          secondary={`Confidence: ${(node.confidence * 100).toFixed(1)}%`}
        />
        {loading ? (
          <CircularProgress size={20} />
        ) : node.hasChildren ? (
          <IconButton edge="end">
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        ) : null}
      </ListItem>
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          {children.map((child) => (
            <TaxonomyNode
              key={child.id}
              node={child}
              level={level + 1}
            />
          ))}
        </List>
      </Collapse>
    </>
  );
};

const TaxonomyViewer = ({ sampleId }) => {
  const [rootNodes, setRootNodes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchTaxonomyData = async () => {
      try {
        const response = await axios.get(`/api/taxonomy/${sampleId}/root`);
        setRootNodes(response.data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch taxonomy data');
        setLoading(false);
      }
    };

    if (sampleId) {
      fetchTaxonomyData();
    }
  }, [sampleId]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={3}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Paper sx={{ width: '100%', maxWidth: 800, mx: 'auto', mt: 2, p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Taxonomy Classification
      </Typography>
      <List>
        {rootNodes.map((node) => (
          <TaxonomyNode key={node.id} node={node} />
        ))}
      </List>
    </Paper>
  );
};

export default TaxonomyViewer;
