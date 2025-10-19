import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.awt.image.BufferedImage;
import java.util.*;
import javax.imageio.ImageIO;
import javax.swing.*;

public class LunarPathfinder extends JFrame {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            LunarPathfinder app = new LunarPathfinder();
            app.setVisible(true);
        });
    }

    private TerrainPanel terrainPanel;
    private ControlPanel controlPanel;

    public LunarPathfinder() {
        setTitle("Lunar Pathfinder - pseudo-3D terrain + A* pathfinding");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        terrainPanel = new TerrainPanel(512, 120, 120); // pixelSize, cols, rows
        controlPanel = new ControlPanel(terrainPanel);

        add(controlPanel, BorderLayout.WEST);
        add(terrainPanel, BorderLayout.CENTER);
        pack();
        setLocationRelativeTo(null);
    }

    private static class ControlPanel extends JPanel {
        private TerrainPanel terrain;
        private JButton regenButton;
        private JButton saveButton;
        private JCheckBox shadingCheck;
        private JSlider slopeWeightSlider;
        private JLabel costLabel;
        private JComboBox<String> sizeCombo;

        ControlPanel(TerrainPanel terrain) {
            this.terrain = terrain;
            setPreferredSize(new Dimension(260, 600));
            setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

            add(new JLabel("<html><h2>Lunar Pathfinder Controls</h2></html>"));
            add(Box.createRigidArea(new Dimension(0, 6)));

            regenButton = new JButton("Regenerate Terrain");
            regenButton.addActionListener(e -> terrain.generateTerrain());
            add(regenButton);
            add(Box.createRigidArea(new Dimension(0, 8)));

            saveButton = new JButton("Save Image (PNG)");
            saveButton.addActionListener(e -> terrain.saveImage());
            add(saveButton);
            add(Box.createRigidArea(new Dimension(0, 8)));

            shadingCheck = new JCheckBox("Enable Shading (pseudo-3D)", true);
            shadingCheck.addActionListener(e -> {
                terrain.setShadingEnabled(shadingCheck.isSelected());
            });
            add(shadingCheck);
            add(Box.createRigidArea(new Dimension(0, 8)));

            add(new JLabel("Slope weight (how costly slopes are)"));
            slopeWeightSlider = new JSlider(0, 500, (int)(terrain.getSlopeWeight()*100));
            slopeWeightSlider.setMajorTickSpacing(100);
            slopeWeightSlider.setPaintTicks(true);
            slopeWeightSlider.setPaintLabels(true);
            slopeWeightSlider.addChangeListener(e -> {
                double w = slopeWeightSlider.getValue()/100.0;
                terrain.setSlopeWeight(w);
            });
            add(slopeWeightSlider);
            add(Box.createRigidArea(new Dimension(0, 12)));

            add(new JLabel("Grid resolution"));
            sizeCombo = new JComboBox<>(new String[] {"40 x 40 (fast)", "80 x 80", "120 x 120 (default)", "200 x 200 (slower)"});
            sizeCombo.setSelectedIndex(2);
            sizeCombo.addActionListener(e -> {
                String sel = (String) sizeCombo.getSelectedItem();
                if (sel.startsWith("40")) terrain.setGridSize(40, 40);
                else if (sel.startsWith("80")) terrain.setGridSize(80, 80);
                else if (sel.startsWith("120")) terrain.setGridSize(120, 120);
                else if (sel.startsWith("200")) terrain.setGridSize(200, 200);
            });
            add(sizeCombo);
            add(Box.createRigidArea(new Dimension(0, 12)));

            add(new JLabel("Info"));
            add(Box.createRigidArea(new Dimension(0, 6)));

            costLabel = new JLabel("Path cost: -");
            terrain.setCostLabel(costLabel);
            add(costLabel);

            add(Box.createVerticalGlue());
            add(new JLabel("<html><small>Click map: first click = start (green). Second click = end (red).</small></html>"));
        }
    }

    private static class TerrainPanel extends JPanel {

        private int cols = 120, rows = 120;
        private int pixelSize = 512;
        private int panelW = 800, panelH = 800;

        private double[][] height; 
        private double minH, maxH;

        private boolean shadingEnabled = true;
        private double elevationScale = 80; 
        private double lightDirX = -0.5, lightDirY = -0.75, lightDirZ = 0.5;

        private Point startCell = null, endCell = null;
        private java.util.List<Point> path = new ArrayList<>();
        private double pathCost = Double.NaN;

        private double slopeWeight = 0.6; 

        private JLabel costLabel = null;

        private long seed = new Random().nextLong();

        TerrainPanel(int pixelSize, int cols, int rows) {
            this.pixelSize = pixelSize;
            this.cols = cols;
            this.rows = rows;
            this.setPreferredSize(new Dimension(panelW, panelH));
            setBackground(Color.BLACK);
            generateTerrain();

            MouseAdapter ma = new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    handleClick(e.getX(), e.getY());
                }
            };
            addMouseListener(ma);

            setPreferredSize(new Dimension(800, 800));
        }

        void setCostLabel(JLabel label) { this.costLabel = label; updateCostLabel(); }

        void setGridSize(int c, int r) {
            this.cols = Math.max(10, c);
            this.rows = Math.max(10, r);
            generateTerrain();
        }

        void setSlopeWeight(double w) {
            this.slopeWeight = w;
            if (startCell != null && endCell != null) {
                computePath();
            }
        }

        double getSlopeWeight() { return slopeWeight; }

        void setShadingEnabled(boolean enabled) {
            this.shadingEnabled = enabled;
            repaint();
        }

        void generateTerrain() {
            seed = new Random().nextLong();
            height = new double[cols][rows];
  
            int octaves = Math.max(3, (int)(Math.log(Math.max(cols, rows))/Math.log(2)));
            double scale = 1.0 / 64.0 * Math.max(1, 120.0/Math.max(cols, rows));
    
            if (cols * rows > 200*200) {
                scale *= 2.0;
            }
            double lacunarity = 2.0;
            double persistence = 0.55;
            PerlinNoise pn = new PerlinNoise(seed);

            minH = Double.POSITIVE_INFINITY;
            maxH = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    double x = i * scale;
                    double y = j * scale;
                    double n = 0;
                    double amp = 1.0, freq = 1.0;
                    for (int o=0;o<octaves;o++) {
                        n += amp * pn.noise(x * freq, y * freq);
                        amp *= persistence;
                        freq *= lacunarity;
                    }
                    
                    double craterEffect = craterLayer(i, j);
                    double val = n*0.5 + craterEffect*0.5;
                    height[i][j] = val;
                    if (val < minH) minH = val;
                    if (val > maxH) maxH = val;
                }
            }
            double range = maxH - minH;
            if (range == 0) range = 1;
            for (int i=0;i<cols;i++) for (int j=0;j<rows;j++) {
                height[i][j] = (height[i][j] - minH) / range;
                height[i][j] += 0.03*(0.5 - pn.noise(i*0.3, j*0.3));
            }
            minH = Double.POSITIVE_INFINITY;
            maxH = Double.NEGATIVE_INFINITY;
            for (int i=0;i<cols;i++) for (int j=0;j<rows;j++) {
                if (height[i][j] < minH) minH = height[i][j];
                if (height[i][j] > maxH) maxH = height[i][j];
            }

            startCell = null; endCell = null;
            path.clear();
            pathCost = Double.NaN;
            updateCostLabel();
            repaint();
        }

        private double craterLayer(int i, int j) {
            Random r = new Random(seed ^ 0x9E3779B97F4A7C15L);
            int craterCount = (int)(Math.max(6, Math.min(30, (cols+rows)/6.0)));
            double accumulator = 0;
            for (int k=0;k<craterCount;k++) {
                double cx = r.nextDouble() * cols;
                double cy = r.nextDouble() * rows;
                double radius = 6 + r.nextDouble()*Math.max(cols,rows)/10.0;
                double dx = i - cx, dy = j - cy;
                double d = Math.sqrt(dx*dx + dy*dy);
                if (d < radius) {
                    double inner = 1.0 - (d/radius);
                    double rim = Math.exp(-Math.pow((d - radius*0.7)/(radius*0.12+0.001), 2));
                    accumulator -= 0.6 * inner; 
                    accumulator += 0.4 * rim;   
                }
            }
            return accumulator;
        }

        void saveImage() {
            BufferedImage img = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2 = img.createGraphics();
            paintComponent(g2);
            g2.dispose();
            try {
                JFileChooser ch = new JFileChooser();
                ch.setSelectedFile(new java.io.File("lunar_terrain.png"));
                if (ch.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                    ImageIO.write(img, "png", ch.getSelectedFile());
                    JOptionPane.showMessageDialog(this, "Saved image.");
                }
            } catch (Exception ex) {
                ex.printStackTrace();
                JOptionPane.showMessageDialog(this, "Failed to save: " + ex.getMessage());
            }
        }

        private void handleClick(int x, int y) {
            Point gridPt = screenToGrid(x, y);
            if (gridPt == null) return;
            if (startCell == null) {
                startCell = gridPt;
                endCell = null;
                path.clear();
                pathCost = Double.NaN;
                updateCostLabel();
            } else if (endCell == null) {
                endCell = gridPt;
                computePath();
            } else {
                // reset and set new start
                startCell = gridPt;
                endCell = null;
                path.clear();
                pathCost = Double.NaN;
                updateCostLabel();
            }
            repaint();
        }

        private void computePath() {
            if (startCell == null || endCell == null) return;
            AStar astar = new AStar(cols, rows, height, slopeWeight);
            AStar.PathResult res = astar.findPath(startCell.x, startCell.y, endCell.x, endCell.y);
            if (res != null) {
                path = res.path;
                pathCost = res.cost;
            } else {
                path.clear();
                pathCost = Double.NaN;
            }
            updateCostLabel();
            repaint();
        }

        private void updateCostLabel() {
            if (costLabel != null) {
                if (Double.isNaN(pathCost)) costLabel.setText("Path cost: -");
                else costLabel.setText(String.format("Path cost: %.3f (lower is faster)", pathCost));
            }
        }

        private Point screenToGrid(int sx, int sy) {
            int bestI=0, bestJ=0;
            double bestDist = Double.POSITIVE_INFINITY;
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    Point2D p = projectCell(i, j);
                    double dx = p.getX() - sx;
                    double dy = p.getY() - sy;
                    double d = dx*dx + dy*dy;
                    if (d < bestDist) { bestDist = d; bestI = i; bestJ = j; }
                }
            }
            return new Point(bestI, bestJ);
        }

        // project grid cell to screen coordinate (center of cell)
        private Point2D projectCell(int i, int j) {
            int w = getWidth(), h = getHeight();
            double cellW = (double) Math.min(w, h) / Math.max(cols, rows) * 0.9;
            double cellH = cellW * 0.5;
            double offsetX = w/2.0;
            double offsetY = h*0.22;

            double baseX = (i - j) * cellW * 0.5;
            double baseY = (i + j) * cellH * 0.5;
            double elev = (height[i][j] - minH) / (maxH - minH + 1e-9);
            double z = elev * elevationScale;
            double screenX = offsetX + baseX;
            double screenY = offsetY + baseY - z;
            return new Point2D.Double(screenX, screenY);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (height == null) return;
            Graphics2D g2 = (Graphics2D) g.create();
            RenderingHints rh = new RenderingHints(
                    RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHints(rh);

            int w = getWidth(), h = getHeight();
            g2.setColor(new Color(18, 18, 20));
            g2.fillRect(0, 0, w, h);

            double cellW = (double) Math.min(w, h) / Math.max(cols, rows) * 0.9;
            double cellH = cellW * 0.5;
            double offsetX = w/2.0;
            double offsetY = h*0.22;

            double lx = lightDirX, ly = lightDirY, lz = lightDirZ;
            double lnorm = Math.sqrt(lx*lx + ly*ly + lz*lz);
            lx/=lnorm; ly/=lnorm; lz/=lnorm;
            int drawOrder = cols+rows;
            for (int s = 0; s < drawOrder; s++) {
                for (int i = 0; i < cols; i++) {
                    int j = s - i;
                    if (j < 0 || j >= rows) continue;
                    if (i+1 >= cols || j+1 >= rows) continue;
                    Point2D p00 = projectCell(i,j);
                    Point2D p10 = projectCell(i+1,j);
                    Point2D p11 = projectCell(i+1,j+1);
                    Point2D p01 = projectCell(i,j+1);

                    Polygon poly = new Polygon();
                    poly.addPoint((int)Math.round(p00.getX()), (int)Math.round(p00.getY()));
                    poly.addPoint((int)Math.round(p10.getX()), (int)Math.round(p10.getY()));
                    poly.addPoint((int)Math.round(p11.getX()), (int)Math.round(p11.getY()));
                    poly.addPoint((int)Math.round(p01.getX()), (int)Math.round(p01.getY()));

                    Vector3 v00 = to3D(i,j);
                    Vector3 v10 = to3D(i+1,j);
                    Vector3 v01 = to3D(i,j+1);
                    Vector3 a = v10.subtract(v00);
                    Vector3 b = v01.subtract(v00);
                    Vector3 normal = a.cross(b).normalized();

                    double diff = Math.max(0, normal.dot(new Vector3(lx, ly, lz)));
                    double avgH = 0.25*(height[i][j] + height[i+1][j] + height[i+1][j+1] + height[i][j+1]);
                    Color base = heightToColor(avgH);
                    Color finalC;
                    if (shadingEnabled) {
                        finalC = blendWithLight(base, diff);
                    } else {
                        finalC = base;
                    }

                    g2.setColor(finalC);
                    g2.fill(poly);
                    g2.setColor(new Color(0,0,0,30));
                    g2.draw(poly);
                }
            }

            if (startCell != null) {
                Point2D ps = projectCell(startCell.x, startCell.y);
                drawMarker(g2, ps, Color.GREEN, "S");
            }
            if (endCell != null) {
                Point2D pe = projectCell(endCell.x, endCell.y);
                drawMarker(g2, pe, Color.RED, "E");
            }

            if (path != null && !path.isEmpty()) {
                Stroke old = g2.getStroke();
                g2.setStroke(new BasicStroke(3.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                g2.setColor(new Color(255, 200, 0, 220));
                Path2D.Double polyline = new Path2D.Double();
                boolean first = true;
                for (Point cell : path) {
                    Point2D p = projectCell(cell.x, cell.y);
                    if (first) {
                        polyline.moveTo(p.getX(), p.getY());
                        first = false;
                    } else polyline.lineTo(p.getX(), p.getY());
                }
                g2.draw(polyline);

                g2.setStroke(new BasicStroke(1.8f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                g2.setColor(new Color(255, 250, 180, 200));
                g2.draw(polyline);
                g2.setStroke(old);
            }

            g2.setColor(Color.WHITE);
            g2.drawString("Grid: " + cols + " x " + rows + "   Seed: " + seed, 10, 18);
            if (startCell != null) g2.drawString("Start: ["+startCell.x+","+startCell.y+"]", 10, 36);
            if (endCell != null) g2.drawString("End: ["+endCell.x+","+endCell.y+"]", 10, 54);
            if (!Double.isNaN(pathCost)) g2.drawString(String.format("Path cost: %.3f", pathCost), 10, 72);

            g2.dispose();
        }

        private void drawMarker(Graphics2D g2, Point2D p, Color color, String label) {
            int r = 8;
            g2.setColor(new Color(0,0,0,160));
            g2.fillOval((int)p.getX()-r-1, (int)p.getY()-r-1, 2*r+2, 2*r+2);
            g2.setColor(color);
            g2.fillOval((int)p.getX()-r, (int)p.getY()-r, 2*r, 2*r);
            g2.setColor(Color.WHITE);
            g2.setFont(g2.getFont().deriveFont(Font.BOLD, 12f));
            g2.drawString(label, (int)p.getX()-4, (int)p.getY()+4);
        }

        private Color heightToColor(double h) {
            int g = (int)Math.round(40 + 200*h);
            g = Math.max(10, Math.min(240, g));
            return new Color(g, g, g);
        }

        private Color blendWithLight(Color base, double diff) {
            double ambient = 0.28;
            double spec = 0.0;
            double bright = ambient + diff*(1-ambient) + spec;
            bright = Math.max(0, Math.min(1, bright));
            int r = (int)Math.round(base.getRed()*bright);
            int gg = (int)Math.round(base.getGreen()*bright);
            int b = (int)Math.round(base.getBlue()*bright);
            return new Color(clamp(r,0,255), clamp(gg,0,255), clamp(b,0,255));
        }

        private int clamp(int v, int a, int b) { return Math.max(a, Math.min(b, v)); }

        private Vector3 to3D(int i, int j) {
            double cellW = (double) Math.min(getWidth(), getHeight()) / Math.max(cols, rows) * 0.9;
            double cellH = cellW * 0.5;
            double x = (i - j) * cellW * 0.5;
            double y = (i + j) * cellH * 0.5;
            double elev = (height[i][j] - minH) / (maxH - minH + 1e-9);
            double z = elev * elevationScale;
            return new Vector3(x, y, z);
        }
    }

    private static class AStar {

        static class Node implements Comparable<Node> {
            int x,y;
            double g = Double.NaN, f = Double.NaN;
            Node parent;
            Node(int x,int y){this.x=x;this.y=y;}
            public int compareTo(Node o){ return Double.compare(this.f,o.f); }
            public boolean equals(Object o){
                if (!(o instanceof Node)) return false;
                Node n=(Node)o; return n.x==x && n.y==y;
            }
            public int hashCode(){ return Objects.hash(x,y); }
        }

        static class PathResult {
            java.util.List<Point> path;
            double cost;
            PathResult(java.util.List<Point> p, double c) { path = p; cost = c; }
        }

        int cols, rows;
        double[][] height;
        double slopeWeight;

        AStar(int cols, int rows, double[][] height, double slopeWeight) {
            this.cols = cols; this.rows = rows; this.height = height; this.slopeWeight = slopeWeight;
        }

        PathResult findPath(int sx, int sy, int ex, int ey) {
            Node start = new Node(sx, sy);
            Node goal = new Node(ex, ey);
            PriorityQueue<Node> open = new PriorityQueue<>();
            Map<Point, Node> allNodes = new HashMap<>();
            start.g = 0;
            start.f = heuristic(sx, sy, ex, ey);
            open.add(start);
            allNodes.put(new Point(sx, sy), start);
            boolean[][] closed = new boolean[cols][rows];

            while (!open.isEmpty()) {
                Node cur = open.poll();
                if (cur.x == ex && cur.y == ey) {
                    java.util.List<Point> path = new ArrayList<>();
                    Node n = cur;
                    while (n != null) {
                        path.add(0, new Point(n.x, n.y));
                        n = n.parent;
                    }
                    return new PathResult(path, cur.g);
                }
                if (closed[cur.x][cur.y]) continue;
                closed[cur.x][cur.y] = true;
                for (int dx=-1; dx<=1; dx++) {
                    for (int dy=-1; dy<=1; dy++) {
                        if (dx==0 && dy==0) continue;
                        int nx = cur.x + dx, ny = cur.y + dy;
                        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
                        if (closed[nx][ny]) continue;
                        double horiz = Math.hypot(dx, dy);
                        double dh = height[nx][ny] - height[cur.x][cur.y];
                        double slope = Math.abs(dh) / (horiz + 1e-12); 
                        double stepCost = horiz * (1.0 + slopeWeight * slope);
                        double tentativeG = cur.g + stepCost;
                        Point key = new Point(nx, ny);
                        Node neighbor = allNodes.get(key);
                        if (neighbor == null) {
                            neighbor = new Node(nx, ny);
                            allNodes.put(key, neighbor);
                        }
                        if (Double.isNaN(neighbor.g) || tentativeG < neighbor.g) {
                            neighbor.g = tentativeG;
                            neighbor.f = tentativeG + heuristic(nx, ny, ex, ey);
                            neighbor.parent = cur;
                            open.remove(neighbor);
                            open.add(neighbor);
                        }
                    }
                }
            }
            return null;
        }

        private double heuristic(int x, int y, int ex, int ey) {
            double dx = ex - x, dy = ey - y;
            double horiz = Math.hypot(dx, dy);
            double dv = Math.abs(height[ex][ey] - height[x][y]);
            double slopeApprox = dv / (1 + horiz);
            return horiz * (1.0 + slopeWeight * slopeApprox);
        }
    }

    private static class PerlinNoise {
        private final int[] perm = new int[512];
        private final Random rnd;
        PerlinNoise(long seed) {
            rnd = new Random(seed);
            int[] p = new int[256];
            for (int i=0;i<256;i++) p[i]=i;
            for (int i=255;i>0;i--) {
                int idx = rnd.nextInt(i+1);
                int tmp = p[i]; p[i] = p[idx]; p[idx] = tmp;
            }
            for (int i=0;i<512;i++) perm[i]=p[i & 255];
        }

        double noise(double x, double y) {
            int X = fastfloor(x) & 255;
            int Y = fastfloor(y) & 255;
            x -= fastfloor(x);
            y -= fastfloor(y);
            double u = fade(x);
            double v = fade(y);
            int aa = perm[X + perm[Y]];
            int ab = perm[X + perm[Y+1]];
            int ba = perm[X+1 + perm[Y]];
            int bb = perm[X+1 + perm[Y+1]];
            double res = lerp(v,
                    lerp(u, grad(aa, x, y), grad(ba, x-1, y)),
                    lerp(u, grad(ab, x, y-1), grad(bb, x-1, y-1))
            );
            return res;
        }

        private static int fastfloor(double x) { return (int)Math.floor(x); }
        private static double fade(double t) { return t*t*t*(t*(t*6-15)+10); }
        private static double lerp(double t, double a, double b) { return a + t*(b-a); }
        private double grad(int hash, double x, double y) {
            int h = hash & 7; 
            double u = h<4 ? x : y;
            double v = h<4 ? y : x;
            return ((h&1)==0 ? u : -u) + ((h&2)==0 ? v : -v);
        }
    }

    private static class Vector3 {
        double x,y,z;
        Vector3(double x,double y,double z){this.x=x;this.y=y;this.z=z;}
        Vector3 subtract(Vector3 o){return new Vector3(x-o.x,y-o.y,z-o.z);}
        Vector3 cross(Vector3 o){
            return new Vector3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x);
        }
        double dot(Vector3 o){return x*o.x + y*o.y + z*o.z;}
        Vector3 normalized(){
            double m = Math.sqrt(x*x+y*y+z*z);
            if (m==0) return new Vector3(0,0,1);
            return new Vector3(x/m,y/m,z/m);
        }
    }
}


