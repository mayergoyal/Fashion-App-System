import React from "react";
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ScrollView,
} from "react-native";

export default function Home({ navigation }) {
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.section}>
        <Text style={styles.mainheading}>Styleकरो</Text>
        <Text style={styles.text}>
          This AI-powered platform helps you explore fashion trends and
          personalize your style.
        </Text>
      </View>

      {/* Buttons */}
      <View style={styles.section}>
        <Text style={styles.heading}>Explore Features</Text>

        <TouchableOpacity
          style={styles.featureButton}
          onPress={() => navigation.navigate("FashionRec")}
        >
          <Image
            source={require("../assets/fashionrec.png")}
            style={styles.featureImg}
          />
          <Text style={styles.buttonText}>Fashion Recommendation</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.featureButton}
          onPress={() => navigation.navigate("TryOut")}
        >
          <Image
            source={require("../assets/try.png")}
            style={styles.featureImg}
          />
          <Text style={styles.buttonText}>Try Out</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, padding: 20, backgroundColor: "#fff" },
  mainheading: { fontSize: 50, margin: 16, textAlign: "center" },
  section: {
    marginBottom: 30,
    padding: 15,
    backgroundColor: "#f9f9f9",
    borderRadius: 10,
  },
  heading: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 10,
  },
  text: { fontSize: 16, color: "#555" },
  featureButton: {
    marginTop: 10,
    padding: 15,
    backgroundColor: "#007bff",
    borderRadius: 10,
    alignItems: "center",
  },
  featureImg: { width: 280, height: 200, marginBottom: 5 },
  buttonText: { fontSize: 18, color: "#fff", fontWeight: "bold" },
});
