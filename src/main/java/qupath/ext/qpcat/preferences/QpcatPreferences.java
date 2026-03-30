package qupath.ext.qpcat.preferences;

import javafx.beans.property.*;
import javafx.collections.ObservableList;
import qupath.fx.prefs.controlsfx.PropertyItemBuilder;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;

/**
 * Persistent preferences for the QP-CAT extension.
 * All preferences are stored using QuPath's preference system and persist across sessions.
 */
public final class QpcatPreferences {

    private static final String CATEGORY_VAE = "QP-CAT: Autoencoder (Advanced)";
    private static final String CATEGORY_GENERAL = "QP-CAT";

    private QpcatPreferences() {}

    // ==================== Autoencoder Training ====================

    private static final IntegerProperty aeLatentDim = PathPrefs.createPersistentPreference(
            "qpcat.ae.latentDim", 16);

    private static final IntegerProperty aeEpochs = PathPrefs.createPersistentPreference(
            "qpcat.ae.epochs", 100);

    private static final DoubleProperty aeLearningRate = PathPrefs.createPersistentPreference(
            "qpcat.ae.learningRate", 0.001);

    private static final IntegerProperty aeBatchSize = PathPrefs.createPersistentPreference(
            "qpcat.ae.batchSize", 128);

    private static final DoubleProperty aeSupervisionWeight = PathPrefs.createPersistentPreference(
            "qpcat.ae.supervisionWeight", 1.0);

    private static final DoubleProperty aeValSplit = PathPrefs.createPersistentPreference(
            "qpcat.ae.valSplit", 0.2);

    private static final IntegerProperty aeEarlyStopPatience = PathPrefs.createPersistentPreference(
            "qpcat.ae.earlyStopPatience", 15);

    private static final BooleanProperty aeClassWeights = PathPrefs.createPersistentPreference(
            "qpcat.ae.classWeights", true);

    private static final BooleanProperty aeAugmentation = PathPrefs.createPersistentPreference(
            "qpcat.ae.augmentation", true);

    private static final StringProperty aeInputMode = PathPrefs.createPersistentPreference(
            "qpcat.ae.inputMode", "measurements");

    private static final IntegerProperty aeTileSize = PathPrefs.createPersistentPreference(
            "qpcat.ae.tileSize", 32);

    private static final BooleanProperty aeIncludeMask = PathPrefs.createPersistentPreference(
            "qpcat.ae.includeMask", true);

    private static final DoubleProperty aeDownsample = PathPrefs.createPersistentPreference(
            "qpcat.ae.downsample", 1.0);

    private static final StringProperty aeNormalization = PathPrefs.createPersistentPreference(
            "qpcat.ae.normalization", "zscore");

    // ==================== Label Sources ====================

    private static final BooleanProperty aeLabelFromLockedAnnotations = PathPrefs.createPersistentPreference(
            "qpcat.ae.labelFromLockedAnnotations", true);

    private static final BooleanProperty aeLabelFromPoints = PathPrefs.createPersistentPreference(
            "qpcat.ae.labelFromPoints", true);

    private static final BooleanProperty aeLabelFromDetections = PathPrefs.createPersistentPreference(
            "qpcat.ae.labelFromDetections", false);

    private static final BooleanProperty aeCellsOnly = PathPrefs.createPersistentPreference(
            "qpcat.ae.cellsOnly", false);

    // ==================== Advanced VAE Training ====================

    private static final DoubleProperty aeKlBetaMax = PathPrefs.createPersistentPreference(
            "qpcat.ae.klBetaMax", 0.5);

    private static final IntegerProperty aeKlCycles = PathPrefs.createPersistentPreference(
            "qpcat.ae.klCycles", 4);

    private static final DoubleProperty aeKlRampFraction = PathPrefs.createPersistentPreference(
            "qpcat.ae.klRampFraction", 0.8);

    private static final DoubleProperty aeFreeBits = PathPrefs.createPersistentPreference(
            "qpcat.ae.freeBits", 0.25);

    private static final DoubleProperty aePretrainFraction = PathPrefs.createPersistentPreference(
            "qpcat.ae.pretrainFraction", 0.1);

    private static final DoubleProperty aeAugNoise = PathPrefs.createPersistentPreference(
            "qpcat.ae.augNoise", 0.02);

    private static final DoubleProperty aeAugScale = PathPrefs.createPersistentPreference(
            "qpcat.ae.augScale", 0.1);

    private static final DoubleProperty aeAugDropout = PathPrefs.createPersistentPreference(
            "qpcat.ae.augDropout", 0.1);

    private static final DoubleProperty aeGradClipNorm = PathPrefs.createPersistentPreference(
            "qpcat.ae.gradClipNorm", 1.0);

    private static final DoubleProperty aeLrSchedulerFactor = PathPrefs.createPersistentPreference(
            "qpcat.ae.lrSchedulerFactor", 0.5);

    private static final IntegerProperty aeLrSchedulerPatience = PathPrefs.createPersistentPreference(
            "qpcat.ae.lrSchedulerPatience", 10);

    private static final IntegerProperty aePointMatchDistance = PathPrefs.createPersistentPreference(
            "qpcat.ae.pointMatchDistance", 50);

    private static final IntegerProperty aeTileBatchSize = PathPrefs.createPersistentPreference(
            "qpcat.ae.tileBatchSize", 500);

    // ==================== Getters / Setters ====================

    public static int getAeLatentDim() { return aeLatentDim.get(); }
    public static void setAeLatentDim(int v) { aeLatentDim.set(v); }

    public static int getAeEpochs() { return aeEpochs.get(); }
    public static void setAeEpochs(int v) { aeEpochs.set(v); }

    public static double getAeLearningRate() { return aeLearningRate.get(); }
    public static void setAeLearningRate(double v) { aeLearningRate.set(v); }

    public static int getAeBatchSize() { return aeBatchSize.get(); }
    public static void setAeBatchSize(int v) { aeBatchSize.set(v); }

    public static double getAeSupervisionWeight() { return aeSupervisionWeight.get(); }
    public static void setAeSupervisionWeight(double v) { aeSupervisionWeight.set(v); }

    public static double getAeValSplit() { return aeValSplit.get(); }
    public static void setAeValSplit(double v) { aeValSplit.set(v); }

    public static int getAeEarlyStopPatience() { return aeEarlyStopPatience.get(); }
    public static void setAeEarlyStopPatience(int v) { aeEarlyStopPatience.set(v); }

    public static boolean isAeClassWeights() { return aeClassWeights.get(); }
    public static void setAeClassWeights(boolean v) { aeClassWeights.set(v); }

    public static boolean isAeAugmentation() { return aeAugmentation.get(); }
    public static void setAeAugmentation(boolean v) { aeAugmentation.set(v); }

    public static String getAeInputMode() { return aeInputMode.get(); }
    public static void setAeInputMode(String v) { aeInputMode.set(v); }

    public static int getAeTileSize() { return aeTileSize.get(); }
    public static void setAeTileSize(int v) { aeTileSize.set(v); }

    public static boolean isAeIncludeMask() { return aeIncludeMask.get(); }
    public static void setAeIncludeMask(boolean v) { aeIncludeMask.set(v); }

    public static double getAeDownsample() { return aeDownsample.get(); }
    public static void setAeDownsample(double v) { aeDownsample.set(v); }

    public static String getAeNormalization() { return aeNormalization.get(); }
    public static void setAeNormalization(String v) { aeNormalization.set(v); }

    public static boolean isAeLabelFromLockedAnnotations() { return aeLabelFromLockedAnnotations.get(); }
    public static void setAeLabelFromLockedAnnotations(boolean v) { aeLabelFromLockedAnnotations.set(v); }

    public static boolean isAeLabelFromPoints() { return aeLabelFromPoints.get(); }
    public static void setAeLabelFromPoints(boolean v) { aeLabelFromPoints.set(v); }

    public static boolean isAeLabelFromDetections() { return aeLabelFromDetections.get(); }
    public static void setAeLabelFromDetections(boolean v) { aeLabelFromDetections.set(v); }

    public static boolean isAeCellsOnly() { return aeCellsOnly.get(); }
    public static void setAeCellsOnly(boolean v) { aeCellsOnly.set(v); }

    /**
     * Saves all current dialog values to persistent preferences.
     */
    public static void saveFromDialog(int latentDim, int epochs, double learningRate,
                                       int batchSize, double supervisionWeight,
                                       double valSplit, int earlyStopPatience,
                                       boolean classWeights, boolean augmentation,
                                       String inputMode, int tileSize, double downsample,
                                       boolean includeMask, String normalization,
                                       boolean labelLocked, boolean labelPoints,
                                       boolean labelDetections, boolean cellsOnly) {
        setAeLatentDim(latentDim);
        setAeEpochs(epochs);
        setAeLearningRate(learningRate);
        setAeBatchSize(batchSize);
        setAeSupervisionWeight(supervisionWeight);
        setAeValSplit(valSplit);
        setAeEarlyStopPatience(earlyStopPatience);
        setAeClassWeights(classWeights);
        setAeAugmentation(augmentation);
        setAeInputMode(inputMode);
        setAeTileSize(tileSize);
        setAeDownsample(downsample);
        setAeIncludeMask(includeMask);
        setAeNormalization(normalization);
        setAeLabelFromLockedAnnotations(labelLocked);
        setAeLabelFromPoints(labelPoints);
        setAeLabelFromDetections(labelDetections);
        setAeCellsOnly(cellsOnly);
    }

    // ==================== Advanced VAE Getters ====================

    public static double getAeKlBetaMax() { return aeKlBetaMax.get(); }
    public static int getAeKlCycles() { return aeKlCycles.get(); }
    public static double getAeKlRampFraction() { return aeKlRampFraction.get(); }
    public static double getAeFreeBits() { return aeFreeBits.get(); }
    public static double getAePretrainFraction() { return aePretrainFraction.get(); }
    public static double getAeAugNoise() { return aeAugNoise.get(); }
    public static double getAeAugScale() { return aeAugScale.get(); }
    public static double getAeAugDropout() { return aeAugDropout.get(); }
    public static double getAeGradClipNorm() { return aeGradClipNorm.get(); }
    public static double getAeLrSchedulerFactor() { return aeLrSchedulerFactor.get(); }
    public static int getAeLrSchedulerPatience() { return aeLrSchedulerPatience.get(); }
    public static int getAePointMatchDistance() { return aePointMatchDistance.get(); }
    public static int getAeTileBatchSize() { return aeTileBatchSize.get(); }

    // ==================== Preferences Pane ====================

    /**
     * Installs QP-CAT preferences into QuPath's Edit > Preferences dialog.
     */
    public static void installPreferences(QuPathGUI qupath) {
        if (qupath == null) return;

        ObservableList<org.controlsfx.control.PropertySheet.Item> items =
                qupath.getPreferencePane().getPropertySheet().getItems();

        // --- KL Annealing ---
        items.add(new PropertyItemBuilder<>(aeKlBetaMax, Double.class)
                .name("KL Beta Max")
                .category(CATEGORY_VAE)
                .description("Maximum KL divergence weight per annealing cycle (default: 0.5). "
                        + "Controls reconstruction vs regularization balance. "
                        + "Lower = better reconstruction, higher = smoother latent space. Range: 0.1-1.0.")
                .build());

        items.add(new PropertyItemBuilder<>(aeKlCycles, Integer.class)
                .name("KL Annealing Cycles")
                .category(CATEGORY_VAE)
                .description("Number of cyclical KL annealing cycles (default: 4). "
                        + "More cycles = more exploration before regularization. "
                        + "Set to 1 for monotonic annealing. Range: 1-10.")
                .build());

        items.add(new PropertyItemBuilder<>(aeKlRampFraction, Double.class)
                .name("KL Ramp Fraction")
                .category(CATEGORY_VAE)
                .description("Fraction of each cycle spent ramping KL from 0 to beta_max (default: 0.8). "
                        + "The remaining fraction holds at beta_max. Range: 0.5-1.0.")
                .build());

        items.add(new PropertyItemBuilder<>(aeFreeBits, Double.class)
                .name("Free Bits (nats)")
                .category(CATEGORY_VAE)
                .description("Minimum KL per latent dimension to prevent posterior collapse (default: 0.25). "
                        + "Higher = stronger anti-collapse but less smooth latent space. Range: 0.0-1.0.")
                .build());

        items.add(new PropertyItemBuilder<>(aePretrainFraction, Double.class)
                .name("Unsupervised Pre-train Fraction")
                .category(CATEGORY_VAE)
                .description("Fraction of epochs to train without classification loss (default: 0.1). "
                        + "Gives the latent space structure before labels are introduced. Range: 0.0-0.5.")
                .build());

        // --- Augmentation ---
        items.add(new PropertyItemBuilder<>(aeAugNoise, Double.class)
                .name("Augmentation Noise Std")
                .category(CATEGORY_VAE)
                .description("Gaussian noise standard deviation for measurement augmentation (default: 0.02). "
                        + "Applied to normalized features. Range: 0.0-0.1.")
                .build());

        items.add(new PropertyItemBuilder<>(aeAugScale, Double.class)
                .name("Augmentation Scale Range")
                .category(CATEGORY_VAE)
                .description("Per-feature random scaling range +/- (default: 0.1 = +/-10%). "
                        + "Simulates staining variability. Range: 0.0-0.3.")
                .build());

        items.add(new PropertyItemBuilder<>(aeAugDropout, Double.class)
                .name("Augmentation Feature Dropout")
                .category(CATEGORY_VAE)
                .description("Probability of zeroing each feature during training (default: 0.1). "
                        + "Improves robustness to missing measurements. Range: 0.0-0.3.")
                .build());

        // --- Training ---
        items.add(new PropertyItemBuilder<>(aeGradClipNorm, Double.class)
                .name("Gradient Clip Max Norm")
                .category(CATEGORY_VAE)
                .description("Maximum gradient norm for clipping (default: 1.0). "
                        + "Prevents exploding gradients. Range: 0.5-5.0.")
                .build());

        items.add(new PropertyItemBuilder<>(aeLrSchedulerFactor, Double.class)
                .name("LR Scheduler Reduction Factor")
                .category(CATEGORY_VAE)
                .description("Factor to reduce learning rate on plateau (default: 0.5 = halve). "
                        + "Range: 0.1-0.9.")
                .build());

        items.add(new PropertyItemBuilder<>(aeLrSchedulerPatience, Integer.class)
                .name("LR Scheduler Patience")
                .category(CATEGORY_VAE)
                .description("Epochs without improvement before reducing LR (default: 10). "
                        + "Range: 5-50.")
                .build());

        // --- Operational ---
        items.add(new PropertyItemBuilder<>(aePointMatchDistance, Integer.class)
                .name("Point Annotation Match Distance (px)")
                .category(CATEGORY_VAE)
                .description("Maximum distance in pixels to match a point annotation to the nearest "
                        + "detection (default: 50). Increase for low-resolution images or large cells.")
                .build());

        items.add(new PropertyItemBuilder<>(aeTileBatchSize, Integer.class)
                .name("Tile I/O Batch Size")
                .category(CATEGORY_VAE)
                .description("Number of tiles read/written per batch during tile-mode training (default: 500). "
                        + "Higher values use more memory but fewer I/O operations. Range: 100-2000.")
                .build());
    }
}
