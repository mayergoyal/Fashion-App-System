import React, { useEffect } from "react";
import { View, Image, StyleSheet } from "react-native";

export default function SplashScreen({ navigation }) {
  useEffect(() => {
    setTimeout(() => {
      navigation.replace("Home"); // Navigate to Home screen after 6 seconds
    }, 6000);
  }, []);

  return (
    <View style={styles.container}>
      <Image source={require("../assets/Homepage.png")} style={styles.image} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  image: { width: "100%", height: "100%", resizeMode: "cover" },
});
