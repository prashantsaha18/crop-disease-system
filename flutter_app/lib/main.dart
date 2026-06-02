// main.dart
// Entry point for the Crop Disease Detector app.
// Sets up the theme, routing, and initial screen.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/home_screen.dart';
import 'screens/result_screen.dart';
import 'screens/disease_info_screen.dart';

// ─────────────────────────────────────────────────────────────────────────── //
//  App colour palette — deep forest greens + warm amber accent
// ─────────────────────────────────────────────────────────────────────────── //

const _primaryGreen    = Color(0xFF1A472A);   // deep forest
const _accentAmber     = Color(0xFFE8A020);   // warm amber
const _surfaceLight    = Color(0xFFF4F7F2);   // off-white tinted green
const _onSurfaceDark   = Color(0xFF1C2117);
const _cardSurface     = Color(0xFFFFFFFF);
const _errorRed        = Color(0xFFD64045);

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait only (typical for plant inspection)
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Transparent system UI for edge-to-edge feel
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      systemNavigationBarColor: _surfaceLight,
      systemNavigationBarIconBrightness: Brightness.dark,
    ),
  );

  runApp(const CropDiseaseApp());
}

class CropDiseaseApp extends StatelessWidget {
  const CropDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Disease Detector',
      debugShowCheckedModeBanner: false,
      theme: _buildTheme(),
      initialRoute: '/',
      onGenerateRoute: _router,
    );
  }

  // ── theme ─────────────────────────────────────────────────────────────── //

  ThemeData _buildTheme() {
    final base = ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.light(
        primary:          _primaryGreen,
        onPrimary:        Colors.white,
        secondary:        _accentAmber,
        onSecondary:      _onSurfaceDark,
        surface:          _cardSurface,
        onSurface:        _onSurfaceDark,
        background:       _surfaceLight,
        onBackground:     _onSurfaceDark,
        error:            _errorRed,
        surfaceVariant:   const Color(0xFFE8EDE4),
        outline:          const Color(0xFFB0BAA8),
      ),
      scaffoldBackgroundColor: _surfaceLight,
    );

    return base.copyWith(
      textTheme: GoogleFonts.soraTextTheme(base.textTheme).copyWith(
        displayLarge: GoogleFonts.sora(
          fontSize: 32, fontWeight: FontWeight.w700,
          color: _onSurfaceDark, letterSpacing: -0.5,
        ),
        displayMedium: GoogleFonts.sora(
          fontSize: 26, fontWeight: FontWeight.w600,
          color: _onSurfaceDark,
        ),
        titleLarge: GoogleFonts.sora(
          fontSize: 20, fontWeight: FontWeight.w600,
          color: _onSurfaceDark,
        ),
        titleMedium: GoogleFonts.sora(
          fontSize: 16, fontWeight: FontWeight.w500,
          color: _onSurfaceDark,
        ),
        bodyLarge: GoogleFonts.sora(
          fontSize: 15, fontWeight: FontWeight.w400,
          color: _onSurfaceDark, height: 1.55,
        ),
        bodyMedium: GoogleFonts.sora(
          fontSize: 13, fontWeight: FontWeight.w400,
          color: const Color(0xFF4A5240), height: 1.5,
        ),
        labelLarge: GoogleFonts.sora(
          fontSize: 14, fontWeight: FontWeight.w600,
          color: _onSurfaceDark, letterSpacing: 0.3,
        ),
        labelSmall: GoogleFonts.sora(
          fontSize: 11, fontWeight: FontWeight.w500,
          color: const Color(0xFF6B7860), letterSpacing: 0.8,
        ),
      ),
      cardTheme: CardTheme(
        color: _cardSurface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: const BorderSide(color: Color(0xFFDEE5D8), width: 1),
        ),
        clipBehavior: Clip.antiAlias,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: _surfaceLight,
        foregroundColor: _onSurfaceDark,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.sora(
          fontSize: 20, fontWeight: FontWeight.w700,
          color: _onSurfaceDark,
        ),
        systemOverlayStyle: const SystemUiOverlayStyle(
          statusBarColor: Colors.transparent,
          statusBarIconBrightness: Brightness.dark,
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _primaryGreen,
          foregroundColor: Colors.white,
          minimumSize: const Size.fromHeight(54),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          elevation: 0,
          textStyle: GoogleFonts.sora(
            fontSize: 15, fontWeight: FontWeight.w600,
          ),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _primaryGreen,
          side: const BorderSide(color: _primaryGreen, width: 1.5),
          minimumSize: const Size.fromHeight(50),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          textStyle: GoogleFonts.sora(
            fontSize: 14, fontWeight: FontWeight.w500,
          ),
        ),
      ),
      chipTheme: ChipThemeData(
        backgroundColor: const Color(0xFFE8EDE4),
        labelStyle: GoogleFonts.sora(
          fontSize: 12, fontWeight: FontWeight.w500,
          color: _primaryGreen,
        ),
        side: BorderSide.none,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      ),
      dividerTheme: const DividerThemeData(
        color: Color(0xFFDEE5D8),
        thickness: 1,
        space: 1,
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        color: _primaryGreen,
        circularTrackColor: Color(0xFFDEE5D8),
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: _onSurfaceDark,
        contentTextStyle: GoogleFonts.sora(color: Colors.white, fontSize: 13),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  // ── routing ───────────────────────────────────────────────────────────── //

  Route<dynamic>? _router(RouteSettings settings) {
    switch (settings.name) {
      case '/':
        return _slide(const HomeScreen(), settings);
      case '/result':
        final args = settings.arguments as ResultScreenArgs;
        return _slide(ResultScreen(args: args), settings);
      case '/diseases':
        return _slide(const DiseaseInfoScreen(), settings);
      default:
        return MaterialPageRoute(
          builder: (_) => const Scaffold(
            body: Center(child: Text('Page not found')),
          ),
        );
    }
  }

  PageRouteBuilder<dynamic> _slide(Widget page, RouteSettings settings) {
    return PageRouteBuilder(
      settings: settings,
      pageBuilder: (_, animation, __) => page,
      transitionsBuilder: (_, animation, __, child) {
        return SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(1, 0),
            end: Offset.zero,
          ).animate(CurvedAnimation(
            parent: animation,
            curve: Curves.easeOutCubic,
          )),
          child: child,
        );
      },
      transitionDuration: const Duration(milliseconds: 320),
    );
  }
}
